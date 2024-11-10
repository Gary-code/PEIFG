CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "log"

IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_BOX_TOKEN = "<box>"

DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

DEFAULT_EXPERT_PAD_TOKEN = '<ref>'
DEFAULT_EXPERT_TOKEN = '<expert>'
DEFAULT_EXPERT_START_TOKEN = '<quad>'
DEFAULT_EXPERT_END_TOKEN = '</quad>'


# ROOT_PATH = '/data/public/ucaswei/data/'

CONVERSATION_DATA = {

    'mydataset': {
        'images': "./vcr1images/",
        'annotations': "./dataset/train.json",
    },
    'mydataset_eval': {
        'images': "./vcr1images/",
        'annotations': "./dataset/test.json",
    }

}