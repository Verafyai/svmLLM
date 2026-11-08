use litesvm::LiteSVM;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use solana_account::Account;
use solana_address::Address;
use solana_compute_budget::compute_budget::ComputeBudget;
use solana_instruction::{account_meta::AccountMeta, Instruction};
use solana_keypair::Keypair;
use solana_message::Message;
use solana_signer::Signer;
use solana_transaction::Transaction;
use std::io::{BufRead, Write};

fn h(s: &str) -> Address { let mut x = Sha256::new(); x.update(s.as_bytes()); Address::new_from_array(x.finalize().into()) }

fn main() {
    let fs = LiteSVM::mainnet_feature_set();
    let mut out = std::io::stdout().lock();
    let feats = json!({
        "virtual_address_space_adjustments": fs.is_active(&agave_feature_set::virtual_address_space_adjustments::id()),
        "account_data_direct_mapping": fs.is_active(&agave_feature_set::account_data_direct_mapping::id()),
    });
    writeln!(out, "{}", json!({"hello": true, "features": feats})).unwrap();
    let program = h("svmtruth:program");
    let payer = Keypair::new();
    let mut cache: Option<(String, LiteSVM)> = None;
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let req: Value = serde_json::from_str(&line).unwrap();
        let elf = req["elf"].as_str().unwrap().to_string();
        if cache.as_ref().map(|c| c.0 != elf).unwrap_or(true) {
            let mut svm = LiteSVM::default().with_feature_set(LiteSVM::mainnet_feature_set()).with_builtins()
                .with_sysvars().with_sigverify(false).with_blockhash_check(false);
            svm.add_program(program, &std::fs::read(&elf).unwrap()).unwrap();
            svm.set_account(payer.pubkey(), Account { lamports: 1_000_000_000_000, data: vec![], owner: Address::new_from_array([0; 32]), executable: false, rent_epoch: 0 }).unwrap();
            cache = Some((elf.clone(), svm));
        }
        let cu = req["cu"].as_u64().unwrap_or(200_000);
        let mut svm = cache.as_ref().unwrap().1.clone().with_compute_budget(ComputeBudget { compute_unit_limit: cu, ..ComputeBudget::new_with_defaults(false) });
        let mut metas = vec![];
        let mut keys = vec![];
        for (i, a) in req["accounts"].as_array().unwrap().iter().enumerate() {
            let data = hex::decode(a["data"].as_str().unwrap()).unwrap();
            let owner = if a["owner"].as_str() == Some("program") { program } else { Address::new_from_array([0; 32]) };
            let key = h(&format!("svmtruth:account:{i}"));
            let lamports = svm.minimum_balance_for_rent_exemption(data.len()).max(1);
            svm.set_account(key, Account { lamports, data, owner, executable: false, rent_epoch: 0 }).unwrap();
            metas.push(if a["writable"].as_bool().unwrap_or(false) { AccountMeta::new(key, false) } else { AccountMeta::new_readonly(key, false) });
            keys.push((key, lamports));
        }
        let data = hex::decode(req["data"].as_str().unwrap()).unwrap();
        let msg = Message::new(&[Instruction { program_id: program, accounts: metas, data }], Some(&payer.pubkey()));
        let tx = Transaction::new(&[&payer], msg, svm.latest_blockhash());
        let reply = match svm.simulate_transaction(tx) {
            Ok(info) => {
                let post: std::collections::BTreeMap<Address, Vec<u8>> = info.post_accounts.into_iter().map(|(k, a)| (k, solana_account::ReadableAccount::data(&a).to_vec())).collect();
                let accts: Vec<String> = keys.iter().map(|(k, _)| hex::encode(post.get(k).cloned().unwrap_or_else(|| svm.get_account(k).unwrap().data))).collect();
                json!({"ok": true, "cu": info.meta.compute_units_consumed, "logs": info.meta.logs, "accounts": accts})
            }
            Err(f) => json!({"ok": false, "err": format!("{:?}", f.err), "cu": f.meta.compute_units_consumed, "logs": f.meta.logs}),
        };
        let mut reply = reply;
        reply["keys"] = json!(keys.iter().map(|(k, l)| json!({"key": hex::encode(k.to_bytes()), "lamports": l})).collect::<Vec<_>>());
        reply["program"] = json!(hex::encode(program.to_bytes()));
        writeln!(out, "{reply}").unwrap();
    }
}
