import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="pathnirvana", meta_file_train="metadata_shuf.csv", path=os.path.join(output_path, "pali_dataset/")
)
audio_config = VitsAudioConfig(
    sample_rate=22050, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0, 
    mel_fmax=None
)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_pali",
    batch_size=48,
    eval_batch_size=32,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner=None, # the text prompts were is already cleaned, remove extra whitespace, special symbols etc
    use_phonemes=False,
    #phoneme_language="en-us",
    #phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    max_audio_len=18 * 22050, # audio longer than this will be ignored
    add_blank=True, # this is by default true for vits, not sure if needed, speed is not changed by much
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        #characters=" !'(),-.:;?ංඃඅආඇඈඉඊඋඌඍඑඒඓඔඕඖකඛගඝඞඟචඡජඣඤඥටඨඩඪණඬතථදධනඳපඵබභමඹයරලවශෂසහළෆ\u0DCA\u0DCF\u0DD0\u0DD1\u0DD2\u0DD3\u0DD4\u0DD6\u0DD8\u0DD9\u0DDA\u0DDB\u0DDC\u0DDD\u0DDE\u0DDF\u0DF2",
        #characters=" !'(),-.:;?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ",
        characters=" !#'(),-.:;?abcdeghijklmnoprstuvxyñāīūḍḷṁṅṇṭ",
        punctuations=" !#'(),-.:;?x",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    test_sentences=[
        ["suppiyassa pana paribbājakassa antevāsī brahmadatto māṇavo anekapariyāyena buddhassa vaṇṇaṁ bhāsati, dhammassa vaṇṇaṁ bhāsati, saṅghassa vaṇṇaṁ bhāsati."],
        ["namo tassa bhagavato arahato sammā sambuddhassa"],
        ["manopubbaṅgamā dhammā manoseṭṭhā manomayā x manasā ce paduṭṭhena bhāsati vā karoti vā x tato naṁ dukkhamanveti, cakkaṁ'va vahato padaṁ."],
        ["mālāgandhavilepanadhāraṇamaṇḍanavibhūsanaṭṭhānā veramaṇīsikkhāpadaṁ samādiyāmi."],
        ["sekhabalasaṅkhittasuttaṁ"],
    ],
    print_step=25,
    print_eval=True,
    mixed_precision=True, # try with false since other multilanguage training was done like that
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    eval_split_max_size=200, # max number of eval samples 
    eval_split_size=0.1, # 10% of the samples to eval
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
