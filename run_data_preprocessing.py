import argparse
import pandas as pd
import json
import random


def write_data(path, data):
    with open(path, "w") as fp:
        for el in data:
            fp.write(json.dumps(el))
            fp.write("\n")


def shuffle_with_mask(arr, mask):
    new_arr = []
    for el in mask:
        new_arr.append(arr[el])
    return new_arr


def random_substring(s):
    pieces = s.split("\n")
    length = len(pieces)
    l = random.randint(0, length // 3)
    r = random.randint(2 * length // 3, length)
    return "\n".join(pieces[l:r])


GOOD_TO_BAD = {
    "O": ["0", "o"],
    "S": ["s"],
    "C": ["c"],
    "g": ["q"],
    "}": ["1"],
    "|": ["I", "l"],
    '"': ["u"],
    "s": ["S"],
    "c": ["C"],
    "0": ["O"],
    ";": [":"],
}


def get_generated_examples(source_code, generated_examples, prob=0.1):
    arr = []
    for _ in range(generated_examples):
        subs = random_substring(source_code)
        new_code = list(subs)
        for i in range(len(new_code)):
            if new_code[i] in GOOD_TO_BAD and random.random() < prob:
                # corrupting
                new_code[i] = random.choice(GOOD_TO_BAD[new_code[i]])
        arr.append((subs, "".join(new_code)))
    return arr


def create_dataset(input_file, output_directory, n_generated_examples, prob, test_sz=0.1):
    df = pd.read_csv(input_file, encoding="utf-16")
    fixed = []
    buggy = []

    for index, row in df.iterrows():
        fixed.append({"code": row["source_code"]})
        buggy.append({"code": row["predicted_code"]})

        if prob > 0:
            examples = get_generated_examples(row["source_code"], n_generated_examples, prob)
            for good, bad in examples:
                fixed.append({"code": good})
                buggy.append({"code": bad})

    data_len = len(fixed)
    shuffle_mask = list(range(data_len))
    random.shuffle(shuffle_mask)
    fixed = shuffle_with_mask(fixed, shuffle_mask)
    buggy = shuffle_with_mask(buggy, shuffle_mask)

    train_path_buggy = f"{output_directory}/train.buggy-fixed.buggy"
    train_path_fixed = f"{output_directory}/train.buggy-fixed.fixed"
    valid_path_buggy = f"{output_directory}/valid.buggy-fixed.buggy"
    valid_path_fixed = f"{output_directory}/valid.buggy-fixed.fixed"
    test_path_buggy = f"{output_directory}/test.buggy-fixed.buggy"
    test_path_fixed = f"{output_directory}/test.buggy-fixed.fixed"

    test_size = int(data_len * test_sz)
    test_with_valid_size = 2 * test_size

    fixed_train = fixed[:-test_with_valid_size]
    buggy_train = buggy[:-test_with_valid_size]
    fixed_valid = fixed[-test_with_valid_size:-test_size]
    buggy_valid = buggy[-test_with_valid_size:-test_size]
    fixed_test = fixed[-test_size:]
    buggy_test = buggy[-test_size:]

    write_data(train_path_fixed, fixed_train)
    write_data(train_path_buggy, buggy_train)
    write_data(valid_path_fixed, fixed_valid)
    write_data(valid_path_buggy, buggy_valid)
    write_data(test_path_fixed, fixed_test)
    write_data(test_path_buggy, buggy_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_directory", type=str)
    parser.add_argument("--n_generated_examples", type=int)
    parser.add_argument("--prob", type=float)
    parser.add_argument("--test_sz", type=float)
    args = parser.parse_args()
    create_dataset(args.input_file, args.output_directory, args.n_generated_examples, args.prob, args.test_sz)


# python3 ./run_data_preprocessing.py --input_file /path/to/paddele_dataset.csv --output_directory \ /some/dir \
# --generated_examples 10 --prob 0.1
if __name__ == "__main__":
    # e = get_generated_examples("UASDpopsd_a---asd@", 10)
    # for el in e:
    #     print(el)
    main()
