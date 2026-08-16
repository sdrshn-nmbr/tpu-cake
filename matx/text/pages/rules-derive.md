# Simple and fast Rust deriving using macro\_rules

July 28, 2025

Reiner Pope and Andrew Pritchard

Rust’s `#[derive]` saves a ton of boilerplate, but custom
derive implementations typically require complicated procedural macros—a
high-friction process that limits adoption. We have released a library
called [rules\_derive](https://github.com/MatX-inc/rules_derive), which
lets you define derivers with much simpler code, using
`macro_rules` instead of procedural macros. We have
experience using it internally for >20 traits, and we no longer use
proc-macro deriving.

Find it on [GitHub](https://github.com/MatX-inc/rules_derive) and [crates.io](https://crates.io/crates/rules_derive).

## Why use rules\_derive?

*Low friction*: Derivers are short—`Clone` takes 40
lines—and can be defined in the same file as they are used in. For
comparison, proc-macro derivers need to be in a separate crate and often
take a few hundred lines.

*High quality*: Derivers handle all tricky cases, like generic
parameters, all struct/enum flavors, and error attribution correctly by
default.

*Composable*: Unlike proc-macros, `macro_rules`
derivers can call other derivers, enabling [some new usage patterns](#advanced-features).

*Fast compiles*: `rules_derive` is 900 lines vs
`syn`/`proc_macro2`/`quote`’s 45k
lines. Clean builds are 6x faster than the standard proc-macro
stack.

## Quick start

Define a deriving macro with `macro_rules!()`:

```
macro_rules! MyTrait {
  (/* accept parsed type definition... */) => {
    // Generate impl
    impl $($generics_bindings)* MyTrait for $ty where $($generics_where)* {
      // implementation here
    }
  }
}
```

Then use it under the `rules_derive` attribute:

```
#[rules_derive(MyTrait)]
struct MyType { x: u32, y: String }
```

The macro definition can be in the same crate or file as its use.

## How it works

`rules_derive` parses type definitions and transforms them
into a uniform format that `macro_rules` can easily process.
Two key transformations make this possible:

### Uniform type syntax

Rust has six syntaxes for defining types, and a deriver should
support all of them:

```
struct NamedFieldsStruct { x: u8, y: u16 }
struct TupleStruct(u8, u16);
struct UnitStruct;
enum NamedFieldsEnum { Variant { x: u8, y: u16 } }
enum TupleEnum { Variant(u8, u16) }
enum UnitEnum { Variant }
```

Among these syntaxes, `NamedFieldsEnum` is the most
general: you can express structs as enums with a single variant, and you
can express types with unnamed fields as types with named fields by
using 0,1,2,… as names.

`rules_derive` converts all of Rust’s type definition
syntaxes to this most general format, and also keeps around the
necessary information to undo that transformation. For example, here is
what `TupleStruct` and `NamedFieldsEnum` get
internally transformed to[1](#fn1):

```
(/* any attributes */)
struct TupleStruct(/* some details elided */)
{
  TupleStruct unnamed (TupleStruct) { 
    field__0 @ 0 : u8,
    field__1 @ 1 : u16,
  }
}

(/* any attributes */)
enum NamedFieldsEnum(/* some details elided */)
{
  Variant named (NamedFieldsEnum::Variant) { 
    field__x @ x : u8,
    field__y @ y : u16, 
  }
}
```

Using `rules_derive`, derivers work with this unified
syntax and automatically handle all six type definition forms. Authors
should always use *named field* syntax inside macros: construct
`TupleStruct` as
`TupleStruct { 0: field__0, 1: field__1 }` rather than
`TupleStruct(field__0, field__1)`.

### Simple generics handling

Correctly handling generic parameters is often a tedious topic for
authors of derivers. Given a type definition

```
struct Foo<T: Clone = u8> where u8: Into<T> { ... }
```

you typically need to generate an instance something like this:

```
impl<T: Clone> SomeTrait for Foo<T> where u8: Into<T> { ... }
```

Note that the original `<T: Clone = u8>` has been
transformed in two ways: first into `<T: Clone>` and
second into `<T>`. Authors of derivers typically need
to parse the generic parameters and create these modified syntax
forms.

Applying these transformations in `macro_rules` is
difficult because of [certain
Rust restrictions](https://doc.rust-lang.org/stable/reference/macros-by-example.html#follow-set-ambiguity-restrictions). To assist, `rules_derive` inserts
parentheses at appropriate places to simplify parsing and provides both
transformed versions of the generic parameters.

For `Foo`, `rules_derive` transforms the
generic parameters to:

```
struct Foo((Foo<T>) (<T: Clone>) where (u8: Into <T>, )) { ... }
```

When defining a deriver you can pick the relevant parts of this
transformed syntax to paste into your `impl` definitions.

### Complete example: Clone deriver

Here’s a complete `Clone` implementation showing the full
pattern:

```
macro_rules! Clone {
  (
    // Accept the parsed type definition (mostly boilerplate)
    ($( ($($attr:tt)*) )*)
    $vis:vis $tystyle:ident $name:ident (($ty:ty)
    ($($generics_bindings:tt)*) where ($($generics_where:tt)*)) {
      $(
        $variant_name:ident ($variant_style:ident $($qualified_variant:tt)*)
            $(= ($discriminant:expr))? {
          $(
            $fieldvis:vis $fieldnameident:ident @ $fieldname:tt : $fieldty:ty,
          )*
        }
      )*
    }
  ) => {
    with_spans! {
      // Generate the impl
      impl $($generics_bindings)* ::std::clone::Clone for $ty where
        $($generics_where)*
        $(
          $(
            // Require all field types implement Clone
            spanned!($fieldty => $fieldty: ::std::clone::Clone,)
          )*
        )*
      {
        fn clone(&self) -> Self {
          match self {
            $(
              // Pattern match and clone each variant
              $($qualified_variant)* { $($fieldname: ($fieldnameident),)* } =>
                $($qualified_variant)* { $($fieldname: $fieldnameident.clone(),)* },
            )*
          }
        }
      }
    }
  }
}
```

For `TupleStruct(u8, u16)`, this generates:

```
impl ::std::clone::Clone for TupleStruct where 
  u8: ::std::clone::Clone, u16: ::std::clone::Clone,
{
  fn clone(&self) -> Self {
    match self {
      TupleStruct { 0: field__0, 1: field__1, } => 
        TupleStruct { 0: field__0.clone(), 1: field__1.clone(), }
    }
  }
}
```

You can see many more complete examples in the [/examples
directory on GitHub](https://github.com/MatX-inc/rules_derive/tree/main/examples).

### Error attribution with `spanned!(...)`

When there is a type error in the code generated by a macro, we want
the Rust compiler to annotate those type errors usefully. In the
`Clone` deriver above, we required that every field’s
type—`$fieldty`—must itself implement `Clone`:

```
$fieldty: ::std::clone::Clone
```

If there is a type error in this code, the error message shouldn’t be
attached to the macro implementation: it should be attached to the field
of the underlying type definition. We use the macro
`spanned!`, also provided by `rules_derive`, to
tell Rust to attribute errors in exactly this way:

```
spanned!($fieldty => $fieldty: ::std::clone::Clone,)
```

## Advanced features

### Composable derivers

`macro_rules` derivers can call other derivers, enabling
new patterns like bundling multiple derivers into one:

```
macro_rules! ValueTypeTraits {
  ($($tt:tt)*) => {
    Copy!($($tt)*);
    Clone!($($tt)*);
    PartialEq!($($tt)*);
    Eq!($($tt)*);
    Debug!($($tt)*);
    Hash!($($tt)*);
  }
}

#[rules_derive(ValueTypeTraits)]
struct Point { x: f32, y: f32 }
```

### External derivers

Proc-macro derivers don’t allow downstream crates to derive traits
for upstream types. For example, `serde` must manually
implement `Serialize` for `std::result::Result`
rather than using `#[derive(Serialize)]`.

`rules_derive` can enable types to expose their
definitions to downstream crates for external derivation. We plan to add
this functionality in a future release.

## What’s next

The first public version of `rules_derive` meets our
internal needs, but has some known limitations that we would like to
improve on:

- Attribute support. To reach approximate parity with proc-macro
  deriving, we’d like to allow macros to define their own attributes, and
  to provide some assistance in parsing attributes. An interesting
  challenge arises in how to detect unused attributes. We have some ideas
  on piggybacking Rust’s dead code warnings to additionally detect unused
  attributes.
- Versioning. If the dependencies of an application use different
  versions of the `rules_derive` library, users currently have
  to be careful to use the correct version of `rules_derive`
  with the appropriate macros. This problem is solvable with some
  work.
- “External” derivers, as discussed above: allow a downstream crate to
  derive traits for upstream types. This could be made easy with some
  work. In particular, we think it would be helpful to offer a crate which
  provides external deriving support for the `std` types with
  public fields, such as `Option`, `Result`, and
  tuples.

We are excited about the possibility of a Rust ecosystem that
provides derivers much more readily than we do today. Check out
rules\_derive [on
GitHub](https://github.com/MatX-inc/rules_derive)!

---

1. The names `field__0`/`field__x`
   are identifiers that macro authors can use when assigning fields to
   variables. This is needed because *fields* can have name
   `0` but *variables* can’t.[⤴](#fnref1)
