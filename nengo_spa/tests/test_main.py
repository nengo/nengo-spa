import os
import os.path

from nengo_spa.main import main


def test_extract_examples(tmpdir):
    main(['nengo_spa', 'extract-examples', os.path.join(
        str(tmpdir), 'extracted')])
    assert len(os.listdir(str(tmpdir))) > 0
