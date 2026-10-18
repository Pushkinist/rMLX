use tokenizers::models::wordlevel::WordLevel;
use tokenizers::Tokenizer;

use super::NeedleWatch;

const NEEDLE: &str = "alpha beta";

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            ["[unk]", "alpha", "beta", "gamma"]
                .iter()
                .enumerate()
                .map(|(i, w)| ((*w).to_owned(), i as u32))
                .collect(),
        )
        .unk_token("[unk]".to_owned())
        .build()
        .expect("word-level model");
    Tokenizer::new(model)
}

#[test]
fn the_watch_fires_on_the_token_that_completes_the_needle() {
    let tk = tokenizer();
    let mut watch = NeedleWatch::new(&tk, NEEDLE);
    assert!(!watch.push(3), "gamma is not the needle");
    assert!(!watch.push(1), "alpha alone is not the needle");
    assert!(watch.push(2), "alpha beta completes it");
    assert!(watch.found());
    assert_eq!(watch.into_ids(), vec![3, 1, 2]);
}

#[test]
fn the_watch_stays_quiet_when_the_needle_never_appears() {
    let tk = tokenizer();
    let mut watch = NeedleWatch::new(&tk, NEEDLE);
    for id in [3, 2, 1, 3, 2] {
        assert!(!watch.push(id));
    }
    assert!(!watch.found());
}

#[test]
fn a_found_needle_stays_found_after_more_tokens() {
    let tk = tokenizer();
    let mut watch = NeedleWatch::new(&tk, NEEDLE);
    for id in [1, 2, 3, 3] {
        watch.push(id);
    }
    assert!(watch.found());
}
