import os
import os.path

from nengo_spa.resources import extract_resource


def test_extract_examples(tmpdir):
    extract_resource('examples', os.path.join(str(tmpdir), 'extracted'))
    assert len(os.listdir(str(tmpdir))) > 0
