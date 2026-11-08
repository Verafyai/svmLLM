//! Syscall exerciser. Instruction data: a sequence of 41-byte ops, each a tag byte and five
//! little-endian u64 arguments, passed to the syscall as given (raw VM addresses included).
//! Each syscall's return value is stored at output[512 + 8 * i]; account 0 is the output.
//! Every syscall takes up to five u64 registers and returns one, so they share one signature.
#![no_std]

extern "C" {
    fn abort(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_panic_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_log_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_log_64_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_log_compute_units_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_log_pubkey(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_log_data(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_memcpy_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_memmove_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_memset_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_memcmp_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_sha256(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_keccak256(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_set_return_data(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_return_data(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_stack_height(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_remaining_compute_units(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_clock_sysvar(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_epoch_schedule_sysvar(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_rent_sysvar(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_epoch_rewards_sysvar(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_last_restart_slot(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_sysvar(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_epoch_stake(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_alloc_free_(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_get_processed_sibling_instruction(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_create_program_address(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_try_find_program_address(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_curve_validate_point(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_curve_group_op(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_curve_multiscalar_mul(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_curve_decompress(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_curve_pairing_map(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_secp256k1_recover(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_poseidon(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_alt_bn128_group_op(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
    fn sol_alt_bn128_compression(a: u64, b: u64, c: u64, d: u64, e: u64) -> u64;
}

/// Tags 0-16 keep their original meaning; see the match in entrypoint.


unsafe fn rd(p: *const u8) -> u64 {
    let mut v = 0u64;
    for i in 0..8 {
        v |= (*p.add(i) as u64) << (8 * i);
    }
    v
}

unsafe fn wr(p: *mut u8, v: u64) {
    for i in 0..8 {
        *p.add(i) = (v >> (8 * i)) as u8;
    }
}

#[no_mangle]
pub unsafe extern "C" fn entrypoint(input: *mut u8) -> u64 {
    // one account, no duplicates: 8 (count) + 88 (header), then data, 10 KiB, alignment, rent epoch
    let dlen = rd(input.add(8 + 80)) as usize;
    let out = input.add(96);
    let mut p = 96 + dlen + 10240;
    p += (8 - p % 8) % 8;
    p += 8;
    let n = rd(input.add(p)) as usize;
    let data = input.add(p + 8);
    let mut i = 0;
    let mut k = 0;
    while i + 41 <= n {
        let t = *data.add(i) as usize;
        let a = |j: usize| rd(data.add(i + 1 + 8 * j));
        let (a0, a1, a2, a3, a4) = (a(0), a(1), a(2), a(3), a(4));
        let r = match t {
            0 => sol_memmove_(a0, a1, a2, a3, a4),
            1 => sol_memcmp_(a0, a1, a2, a3, a4),
            2 => sol_memcpy_(a0, a1, a2, a3, a4),
            3 => sol_memset_(a0, a1, a2, a3, a4),
            4 => sol_log_(a0, a1, a2, a3, a4),
            5 => sol_log_64_(a0, a1, a2, a3, a4),
            6 => sol_log_compute_units_(a0, a1, a2, a3, a4),
            7 => sol_panic_(a0, a1, a2, a3, a4),
            8 => sol_sha256(a0, a1, a2, a3, a4),
            9 => sol_keccak256(a0, a1, a2, a3, a4),
            10 => sol_log_pubkey(a0, a1, a2, a3, a4),
            11 => sol_log_data(a0, a1, a2, a3, a4),
            12 => sol_set_return_data(a0, a1, a2, a3, a4),
            13 => sol_get_return_data(a0, a1, a2, a3, a4),
            14 => sol_get_stack_height(a0, a1, a2, a3, a4),
            15 => sol_remaining_compute_units(a0, a1, a2, a3, a4),
            16 => abort(a0, a1, a2, a3, a4),
            17 => sol_get_clock_sysvar(a0, a1, a2, a3, a4),
            18 => sol_get_epoch_schedule_sysvar(a0, a1, a2, a3, a4),
            19 => sol_get_rent_sysvar(a0, a1, a2, a3, a4),
            20 => sol_get_epoch_rewards_sysvar(a0, a1, a2, a3, a4),
            21 => sol_get_last_restart_slot(a0, a1, a2, a3, a4),
            22 => sol_get_sysvar(a0, a1, a2, a3, a4),
            23 => sol_get_epoch_stake(a0, a1, a2, a3, a4),
            24 => sol_alloc_free_(a0, a1, a2, a3, a4),
            25 => sol_get_processed_sibling_instruction(a0, a1, a2, a3, a4),
            26 => sol_create_program_address(a0, a1, a2, a3, a4),
            27 => sol_try_find_program_address(a0, a1, a2, a3, a4),
            28 => sol_curve_validate_point(a0, a1, a2, a3, a4),
            29 => sol_curve_group_op(a0, a1, a2, a3, a4),
            30 => sol_curve_multiscalar_mul(a0, a1, a2, a3, a4),
            31 => sol_curve_decompress(a0, a1, a2, a3, a4),
            32 => sol_curve_pairing_map(a0, a1, a2, a3, a4),
            33 => sol_secp256k1_recover(a0, a1, a2, a3, a4),
            34 => sol_poseidon(a0, a1, a2, a3, a4),
            35 => sol_alt_bn128_group_op(a0, a1, a2, a3, a4),
            36 => sol_alt_bn128_compression(a0, a1, a2, a3, a4),
            _ => 0,
        };
        if k < 64 && dlen >= 1024 {
            wr(out.add(512 + 8 * k), r);
        }
        k += 1;
        i += 41;
    }
    0
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        abort(0, 0, 0, 0, 0);
    }
    loop {}
}
