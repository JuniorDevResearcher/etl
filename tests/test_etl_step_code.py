import os


def test_no_table_dataset_assignment():
    for step_file in iter_steps():
        with open(step_file) as istream:
            for i, line in enumerate(istream):
                if ".metadata.dataset = " in line:
                    assert False, "Bad table dataset assignment found in %s:%d" % (step_file, i)


def iter_steps():
    for root, dirs, files in os.walk("etl/steps/data"):
        for filename in files:
            if filename.endswith(".py"):
                yield os.path.join(root, filename)
