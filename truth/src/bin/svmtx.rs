// Ground truth for whole transactions: several programs deployed at chosen addresses (upgradeable
// loader, as LiteSVM's add_program does), accounts with explicit keys, owners, lamports, data and
// signer flags, and one instruction with any account metas (duplicates allowed). Reports every
// listed account before and after, so the replay starts from exactly the same state.
// stdin, one JSON per line:
//   {"cu": n, "programs": [{"id": hex, "elf": path}],
//    "accounts": [{"key": hex, "owner": hex, "lamports": n, "data": hex}],
//    "ix": {"program": hex, "metas": [{"key": hex, "signer": b, "writable": b}], "data": hex}}
use litesvm::LiteSVM;
use serde_json::{json, Value};
use solana_account::{Account, ReadableAccount};
use solana_address::Address;
use solana_compute_budget::compute_budget::ComputeBudget;
use solana_instruction::{account_meta::AccountMeta, Instruction};
use solana_keypair::Keypair;
use solana_message::Message;
use solana_signer::Signer;
use solana_transaction::Transaction;
use std::io::{BufRead, Write};

fn addr(h: &str) -> Address {
    Address::new_from_array(hex::decode(h).unwrap().try_into().unwrap())
}

fn dump(svm: &LiteSVM, k: &Address, a: Option<Account>) -> Value {
    let a = a.or_else(|| svm.get_account(k));
    match a {
        Some(a) => json!({"key": hex::encode(k.to_bytes()), "owner": hex::encode(a.owner.to_bytes()), "lamports": a.lamports,
                          "data": hex::encode(&a.data), "executable": a.executable}),
        None => json!({"key": hex::encode(k.to_bytes()), "missing": true}),
    }
}

fn main() {
    let payer = Keypair::new();
    let mut out = std::io::stdout().lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let req: Value = serde_json::from_str(&line).unwrap();
        let mut svm = LiteSVM::default().with_feature_set(LiteSVM::mainnet_feature_set()).with_builtins()
            .with_sysvars().with_sigverify(false).with_blockhash_check(false)
            .with_compute_budget(ComputeBudget { compute_unit_limit: req["cu"].as_u64().unwrap_or(200_000), ..ComputeBudget::new_with_defaults(false) });
        svm.set_account(payer.pubkey(), Account { lamports: 1_000_000_000_000, data: vec![], owner: Address::new_from_array([0; 32]), executable: false, rent_epoch: 0 }).unwrap();
        for p in req["programs"].as_array().unwrap() {
            svm.add_program(addr(p["id"].as_str().unwrap()), &std::fs::read(p["elf"].as_str().unwrap()).unwrap()).unwrap();
        }
        let mut keys: Vec<Address> = vec![];
        for a in req["accounts"].as_array().unwrap() {
            let k = addr(a["key"].as_str().unwrap());
            svm.set_account(k, Account { lamports: a["lamports"].as_u64().unwrap(), data: hex::decode(a["data"].as_str().unwrap()).unwrap(),
                owner: addr(a["owner"].as_str().unwrap()), executable: false, rent_epoch: 0 }).unwrap();
            keys.push(k);
        }
        let ix = &req["ix"];
        let metas: Vec<AccountMeta> = ix["metas"].as_array().unwrap().iter().map(|m| AccountMeta {
            pubkey: addr(m["key"].as_str().unwrap()), is_signer: m["signer"].as_bool().unwrap(), is_writable: m["writable"].as_bool().unwrap() }).collect();
        let program = addr(ix["program"].as_str().unwrap());
        let mut listed: Vec<Address> = keys.clone();
        for m in &metas { if !listed.contains(&m.pubkey) { listed.push(m.pubkey); } }
        if !listed.contains(&program) { listed.push(program); }
        let pre: Vec<Value> = listed.iter().map(|k| dump(&svm, k, None)).collect();
        let msg = Message::new(&[Instruction { program_id: program, accounts: metas, data: hex::decode(ix["data"].as_str().unwrap()).unwrap() }], Some(&payer.pubkey()));
        let n_sig = msg.header.num_required_signatures as usize;
        let tx = Transaction { signatures: vec![Default::default(); n_sig], message: msg };
        let reply = match svm.simulate_transaction(tx) {
            Ok(info) => {
                let post: std::collections::BTreeMap<Address, Account> = info.post_accounts.into_iter().map(|(k, a)| (k, Account {
                    lamports: a.lamports(), data: a.data().to_vec(), owner: *a.owner(), executable: a.executable(), rent_epoch: 0 })).collect();
                let after: Vec<Value> = listed.iter().map(|k| dump(&svm, k, post.get(k).cloned())).collect();
                json!({"ok": true, "cu": info.meta.compute_units_consumed, "logs": info.meta.logs, "pre": pre, "post": after})
            }
            Err(f) => json!({"ok": false, "err": format!("{:?}", f.err), "cu": f.meta.compute_units_consumed, "logs": f.meta.logs, "pre": pre}),
        };
        writeln!(out, "{reply}").unwrap();
    }
}
