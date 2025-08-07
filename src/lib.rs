#![warn(
    rust_2018_idioms,
    let_underscore_drop,
    clippy::assertions_on_result_states,
    clippy::bool_to_int_with_if,
    clippy::borrow_as_ptr,
    clippy::cloned_instead_of_copied,
    clippy::create_dir,
    clippy::debug_assert_with_mut_call,
    clippy::default_union_representation,
    clippy::deref_by_slicing,
    clippy::derive_partial_eq_without_eq,
    clippy::empty_drop,
    clippy::empty_line_after_outer_attr,
    clippy::empty_structs_with_brackets,
    clippy::equatable_if_let,
    clippy::expl_impl_clone_on_copy,
    clippy::explicit_deref_methods,
    clippy::explicit_into_iter_loop,
    clippy::explicit_iter_loop,
    clippy::filetype_is_file,
    clippy::filter_map_next,
    clippy::flat_map_option,
    clippy::float_cmp_const,
    clippy::format_push_string,
    clippy::from_iter_instead_of_collect,
    clippy::get_unwrap,
    clippy::implicit_clone,
    clippy::implicit_saturating_sub,
    clippy::imprecise_flops,
    clippy::index_refutable_slice,
    clippy::inefficient_to_string,
    clippy::invalid_upcast_comparisons,
    clippy::iter_on_empty_collections,
    clippy::iter_on_single_items,
    clippy::large_types_passed_by_value,
    clippy::let_unit_value,
    clippy::lossy_float_literal,
    clippy::macro_use_imports,
    clippy::manual_assert,
    clippy::manual_clamp,
    clippy::manual_instant_elapsed,
    clippy::manual_let_else,
    clippy::manual_ok_or,
    clippy::manual_string_new,
    clippy::map_unwrap_or,
    clippy::match_bool,
    clippy::match_same_arms,
    clippy::mut_mut,
    clippy::mutex_atomic,
    clippy::mutex_integer,
    clippy::naive_bytecount,
    clippy::needless_bitwise_bool,
    clippy::needless_collect,
    clippy::needless_continue,
    clippy::needless_for_each,
    clippy::nonstandard_macro_braces,
    clippy::or_fun_call,
    clippy::path_buf_push_overwrite,
    clippy::range_minus_one,
    clippy::range_plus_one,
    clippy::redundant_else,
    clippy::rest_pat_in_fully_bound_structs,
    clippy::semicolon_if_nothing_returned,
    clippy::significant_drop_in_scrutinee,
    clippy::str_to_string,
    clippy::string_add,
    clippy::string_add_assign,
    clippy::string_lit_as_bytes,
    clippy::string_to_string,
    clippy::suspicious_to_owned,
    clippy::trait_duplication_in_bounds,
    clippy::trivially_copy_pass_by_ref,
    clippy::type_repetition_in_bounds,
    clippy::unchecked_duration_subtraction,
    clippy::unicode_not_nfc,
    clippy::uninlined_format_args,
    clippy::unnecessary_join,
    clippy::unnecessary_self_imports,
    clippy::unneeded_field_pattern,
    clippy::unnested_or_patterns,
    clippy::unused_peekable,
    clippy::unused_rounding,
    clippy::use_self,
    clippy::used_underscore_binding,
    clippy::useless_let_if_seq
)]

//! An open-source postflop solver library.
//!
//! # Examples
//!
//! See the [examples] directory.
//!
//! [examples]: https://github.com/b-inary/postflop-solver/tree/main/examples
//!
//! # Implementation details
//! - **Algorithm**: The solver uses the state-of-the-art [Discounted CFR] algorithm.
//!   Currently, the value of γ is set to 3.0 instead of the 2.0 recommended in the original paper.
//!   Also, the solver resets the cumulative strategy when the number of iterations is a power of 4.
//! - **Performance**: The solver engine is highly optimized for performance with maintainable code.
//!   The engine supports multithreading by default, and it takes full advantage of unsafe Rust in hot spots.
//!   The developer reviews the assembly output from the compiler and ensures that SIMD instructions are used as much as possible.
//!   Combined with the algorithm described above, the performance surpasses paid solvers such as PioSOLVER and GTO+.
//! - **Isomorphism**: The solver does not perform any abstraction.
//!   However, isomorphic chances (turn and river deals) are combined into one.
//!   For example, if the flop is monotone, the three non-dealt suits are isomorphic,
//!   allowing us to skip the calculation for two of the three suits.
//! - **Precision**: 32-bit floating-point numbers are used in most places.
//!   When calculating summations, temporary values use 64-bit floating-point numbers.
//!   There is also a compression option where each game node stores the values
//!   by 16-bit integers with a single 32-bit floating-point scaling factor.
//! - **Bunching effect**: At the time of writing, this is the only implementation that can handle the bunching effect.
//!   It supports up to four folded players (6-max game).
//!   The implementation correctly counts the number of card combinations and does not rely on heuristics
//!   such as manipulating the probability distribution of the deck.
//!   Note, however, that enabling the bunching effect increases the time complexity
//!   of the evaluation at the terminal nodes and slows down the computation significantly.
//!
//! [Discounted CFR]: https://arxiv.org/abs/1809.04040
//!
//! # Crate features
//! - `bincode`: Uses [bincode] crate (2.0.0-rc.3) to serialize and deserialize the `PostFlopGame` struct.
//!   This feature is required to save and load the game tree.
//!   Enabled by default.
//! - `custom-alloc`: Uses custom memory allocator in solving process (only available in nightly Rust).
//!   It significantly reduces the number of calls of the default allocator,
//!   so it is recommended to use this feature when the default allocator is not so efficient.
//!   Note that this feature assumes that, at most, only one instance of `PostFlopGame` is available
//!   when solving in a program.
//!   Disabled by default.
//! - `rayon`: Uses [rayon] crate for parallelization.
//!   Enabled by default.
//! - `zstd`: Uses [zstd] crate to compress and decompress the game tree.
//!   This feature is required to save and load the game tree with compression.
//!   Disabled by default.
//!
//! [bincode]: https://github.com/bincode-org/bincode
//! [rayon]: https://github.com/rayon-rs/rayon
//! [zstd]: https://github.com/gyscos/zstd-rs

#![cfg_attr(feature = "custom-alloc", feature(allocator_api))]

#[cfg(feature = "custom-alloc")]
mod alloc;

#[cfg(feature = "bincode")]
mod file;

mod action_tree;
mod atomic_float;
mod bet_size;
mod bunching;
mod card;
mod game;
mod hand;
mod hand_table;
mod interface;
mod mutex_like;
mod range;
mod sliceop;
mod solver;
mod utility;

#[cfg(feature = "bincode")]
pub use file::*;

pub use action_tree::*;
pub use bet_size::*;
pub use bunching::*;
pub use card::*;
pub use game::*;
pub use interface::*;
pub use mutex_like::*;
pub use range::*;
pub use solver::*;
pub use utility::*;
