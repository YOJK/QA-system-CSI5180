import csv

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

from nlpaug.util import Action

if __name__ == "__main__":
    text = 'The quick brown fox jumped over the lazy dog'
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en'
    )
    print(back_translation_aug.augment(text))
    aug = naw.RandomWordAug(action="swap")
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)

    aug = naw.RandomWordAug()
    augmented_text = aug.augment(text)
    print("Original:")
    print(text)
    print("Augmented Text:")
    print(augmented_text)
    filename = 'test.csv'
    with open(filename,'rt',encoding='GBK') as f:
        data_corpus = csv.reader(f)
        # row_count = sum(1 for line in open(filename))
        data = list(data_corpus)
        row_count = len(data)
        print(row_count)