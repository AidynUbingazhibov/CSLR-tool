import streamlit as st
import shutil
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
import cv2
import time
import utils.SessionState as SessionState
from random import randint
from streamlit import caching
import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
import copy
from components.custom_slider import custom_slider
from decord import VideoReader
from decord import cpu, gpu
import pandas as pd
from torchvision import transforms
import numpy as np
from PIL import Image
sys.path.append("CSLR/stochastic-cslr")
import stochastic_cslr
import torch
import glob
import time

class LookupTable:
    def __init__(self, words=None, symbols=None, allow_unk=False):
        """
        Args:
            words: all the words, unsorted, allows duplications
            symbols: no duplications
        """
        assert words is None or symbols is None, "Specify either words or symbols."

        self.allow_unk = allow_unk

        if symbols is None:
            symbols = sorted(set(words))

        assert len(symbols) == len(set(symbols)), "Symbols contain duplications."

        self.symbols = symbols
        self.mapping = {symbol: i for i, symbol in enumerate(symbols)}

    def __call__(self, symbol):
        if symbol in self.mapping:
            return self.mapping[symbol]
        elif self.allow_unk:
            return len(self) - 1
        raise KeyError(symbol)

    def __getitem__(self, i):
        if i < len(self.symbols):
            return self.symbols[i]
        elif i == len(self.symbols):
            return "unk"
        else:
            raise IndexError(f"Index {i} out of range.")

    def __len__(self):
        return len(self.mapping) + int(self.allow_unk)

    def __str__(self):
        unk = {"unk": len(self) - 1} if self.allow_unk else {}
        return str({**self.mapping, **unk})


# define recommend values for model confidence and nms suppression

def_values ={'conf': 70, 'nms': 50}

keys = ['conf', 'nms']



@st.cache(

    hash_funcs={

        st.delta_generator.DeltaGenerator: lambda x: None,

        "_regex.Pattern": lambda x: None,

    },

    allow_output_mutation=True,

)


def sample_indices(n, p_drop, random_drop):
    p_kept = 1 - p_drop

    if random_drop:
        indices = np.arange(n)
        np.random.shuffle(indices)
        indices = indices[: int(n * p_kept)]
        indices = sorted(indices)
    else:
        indices = np.arange(0, n, 1 / p_kept)
        indices = np.round(indices)
        indices = np.clip(indices, 0, n - 1)
        indices = indices.astype(int)
    return indices

def get_frames(video_path):
    # frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
    vr = VideoReader(video_path, ctx=cpu(0))
    return vr

def load_data_frame(split, sep):
    """Load corpus."""
    path = f"/app/CSLR/stochastic-cslr/{split}.csv"
    df = pd.read_csv(path, sep = sep)
    df["annotation"] = df["annotation"].apply(str.split)

    return df

def create_vocab(split, sep):
    df = load_data_frame(split, sep)
    sentences = df["annotation"].to_list()
    return LookupTable(
        [gloss for sentence in sentences for gloss in sentence],
        allow_unk=True,
    )

def sup(preds):
    res = []
    for i in range(len(preds)):
        if preds[i] == 0 or (i > 0 and preds[i] == preds[i - 1]):
            continue
        res.append(preds[i])
    return res

def inference(epoch, vocab, frames, lang):
    model = stochastic_cslr.load_model(1, epoch=epoch, lang=lang)
    #model = stochastic_cslr.load_model()
    model.to("cpu")
    model.eval()
    lpis = model([torch.tensor(frames).to("cpu")])
    prob = []

    prob += [lpi.exp().detach().cpu().numpy() for lpi in lpis]
    hyp = model.decode(prob=prob, beam_width=10, prune=0.01, nj=4)
    print([" ".join([vocab[i] for i in hi]) for hi in hyp])
    hyp = [sup(h) for h in hyp]
    hyp = [" ".join([vocab[i] for i in hi]) for hi in hyp]

    return hyp

def parameter_sliders(key, enabled = True, value = None):

    conf = custom_slider("Model Confidence",

                        minVal = 0, maxVal = 100, InitialValue= value[0], enabled = enabled,

                        key = key[0])

    nms = custom_slider('Overlapping Threshold',

                        minVal = 0, maxVal = 100, InitialValue= value[1], enabled = enabled,

                        key = key[1])



    return(conf, nms)

def trigger_rerun():

    """

    mechanism in place to force resets and update widget states

    """

    session_infos = Server.get_current()._session_info_by_id.values()

    for session_info in session_infos:

        this_session = session_info.session

    this_session.request_rerun()

def main():
    #st.set_page_config(page_title = "Continuous Sign Language Recognition")
    st.markdown("### Model Architecture")

    st.image(
        f'/app/architecture.png',
        caption='Architecture overview',
        use_column_width=True
    )

    base_size = [256, 256]
    crop_size = [224, 224]
    random_crop = False
    p_drop = 0.5
    random_drop = False

    transform_phoenix = transforms.Compose(
    [
        transforms.Resize(base_size),
        transforms.RandomCrop(crop_size)
        if random_crop
        else transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Normalize([0.53724027, 0.5272855, 0.51954997], [1, 1, 1])
    ]
    )

    transform_krsl = transforms.Compose(
    [
        transforms.Resize(base_size),
        transforms.RandomCrop(crop_size)
        if random_crop
        else transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Normalize([0.53724027, 0.5272855, 0.51954997], [1, 1, 1])
    ]
    )


    state = SessionState.get(upload_key = None, enabled = True,
    start = False, conf = 70, nms = 50, run = False, upload_db = False)
    hide_streamlit_widgets()

    """

    # Continuous Sign Language Recogntion

    """

    with open("/app/phrases.txt", "r") as f:
        lines = f.readlines()

    my_phrases = [""] + [line.strip().split("\t")[1] for line in lines]

    with open("/app/app/test_ids.txt", "r") as f:
        ids = f.readlines()

    signer_ids = [""] + [id.strip() for id in ids]
    phrase_dict = {line.strip().split("\t")[1]:line.strip().split("\t")[0] for line in lines}

    with st.sidebar:
        """

        ## :floppy_disk: Stochastic CSLR model
        SOTA among single cue models


        """

        #state.conf, state.nms = parameter_sliders(

        #    keys, state.enabled, value = [state.conf, state.nms])

        st.text("")

        st.text("")

        st.text("")

        lang = st.radio("Select language: ", ('Russian', 'German'))

        backbone = st.sidebar.selectbox(
            label = 'Please choose the backbone for Stochastic CSLR',

            options = [
                'ResNet18'
            ],

            index = 0,

            key = 'backbone'

        )

        phrase = st.sidebar.selectbox(
            label = "Please select the phrase for K-RSL dataset here",

            options = my_phrases,

            index = 0,

            key = 'phrase'
        )

        signer_id = st.sidebar.selectbox(
            label = "Please select the signer id for K-RSL dataset here",

            options = signer_ids,

            index = 0,

            key = 'signer_id'
        )

    upload = st.empty()
    start_button = st.empty()
    stop_button = st.empty()

    with upload:
        f = st.file_uploader('Upload Video file (mpeg/mp4 format)', key = state.upload_key)

    if lang == "Russian" and len(phrase) != 0 and len(signer_id) != 0:
        video_path = "/app/test_videos/" + str(phrase_dict[phrase]) + "/" + "P" + str(signer_id) + "_" + "S" + str(phrase_dict[phrase]) + "_" + "00.mp4"

        if not os.path.exists(video_path):
            st.info("The video is not in the database!")
            return

        vf = cv2.VideoCapture(video_path)
        vf = cv2.VideoCapture(video_path)
        frames = get_frames(video_path=video_path)
        indices = sample_indices(n=len(frames), p_drop=p_drop, random_drop=random_drop)
        frames = [Image.fromarray(frames[i].asnumpy(), 'RGB') for i in indices]

        if lang == "Russian":
            frames = map(transform_krsl, frames)
        else:
            frames = map(transform_phoenix, frames)

        frames = np.stack(list(frames))

        if lang == "Russian":
            epoch = 18
            vocab = create_vocab(split="train_rus", sep=",")
        else:
            vocab = create_vocab(split="train_ger", sep="|")

            if backbone == "ResNet18":
                epoch = 100
            else:
                epoch = 200

        hyp = inference(epoch, vocab, frames, lang)

        if not state.run:
            start_button.empty()
            start = start_button.button("PREDICT")
            state.start = start

        if state.start:
            start_button.empty()
            state.enabled = False

            if state.run:
                if phrase in phrase_dict:
                    phrase_id = phrase_dict[phrase]

                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                ProcessFrames(vf, stop_button, hyp, video_path, phrase_id, signer_ids, state)
            else:
                state.run = True
                trigger_rerun()


    if f is not None:
        tfile  = tempfile.NamedTemporaryFile(delete = False)
        tfile.write(f.read())

        upload.empty()
        vf = cv2.VideoCapture(tfile.name)
        frames = get_frames(video_path=tfile.name)
        indices = sample_indices(n=len(frames), p_drop=p_drop, random_drop=random_drop)
        frames = [Image.fromarray(frames[i].asnumpy(), 'RGB') for i in indices]

        if lang == "Russian":
            frames = map(transform_krsl, frames)
        else:
            frames = map(transform_phoenix, frames)

        frames = np.stack(list(frames))

        if lang == "Russian":
            epoch = 18
            vocab = create_vocab(split="train_rus", sep=",")
        else:
            vocab = create_vocab(split="train_ger", sep="|")

            if backbone == "ResNet18":
                epoch = 100
            else:
                epoch = 200

        hyp = inference(epoch, vocab, frames, lang)

        if not state.run:
            start_button.empty()
            start = start_button.button("PREDICT ")
            state.start = start

            with open("/app/app/upload.txt") as f:
                bool = int(f.readline())
            phrase_id = None
            if phrase in phrase_dict:
                    phrase_id = phrase_dict[phrase]

            if bool and phrase_id != None:
                up = upload.button("UPLOAD TO DATABASE")
                state.upload_db = up

        if state.upload_db:
            with open("/app/app/test_ids.txt", "a") as f:
                if "51" not in signer_ids:
                    f.write("51\n")

                shutil.move(tfile.name, f"/app/test_videos/{phrase_id}/P51_S{phrase_id}_00.mp4")
                st.info("The data was successfully uploaded to the database!")
            state.run = False



        if state.start:
            start_button.empty()
            state.enabled = False

            if state.run:
                f.close()
                state.upload_key = str(randint(1000, int(1e6)))
                state.enabled = True
                state.run = False
                phrase_id = None

                if phrase in phrase_dict:
                    phrase_id = phrase_dict[phrase]

                ProcessFrames(vf, stop_button, hyp, tfile, phrase_id, signer_ids, state)
            else:
                state.run = True
                trigger_rerun()


def hide_streamlit_widgets():

    """

    hides widgets that are displayed by streamlit when running

    """

    hide_streamlit_style = """

            <style>

            #MainMenu {visibility: hidden;}

            footer {visibility: hidden;}

            </style>

            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def ProcessFrames(vf, stop, hyp, tfile, phrase_id, signer_ids, state):
    """

        main loop for processing video file:

        Params

        vf = VideoCapture Object

        tracker = Tracker Object that was instantiated 

        obj_detector = Object detector (model and some properties) 

    """

    try:
        num_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        print('Total number of frames to be processed:', num_frames,
        '\nFrame rate (frames per second):', fps)
    except:
        print('We cannot determine number of frames and FPS!')

    frame_counter = 0
    _stop = stop.button("stop")
    new_car_count_txt = st.empty()
    fps_meas_txt = st.empty()
    bar = st.progress(frame_counter)
    stframe = st.empty()
    start = time.time()
    pred_txt = st.empty()
    upload = st.empty()

    while vf.isOpened():
        # if frame is read correctly ret is True

        ret, frame = vf.read()
        if _stop:
            break

        if not ret:
            st.markdown("""

                <style>

                .big-font {

                font-size:25px !important;

              }

              </style>

              """, unsafe_allow_html=True)

            st.markdown(f'**Prediction**: <p class="big-font">{hyp[0]} </p>', unsafe_allow_html=True)

            with open("/app/app/upload.txt") as f:
                bool = int(f.readline())

            if phrase_id == None or not bool:
                st.info("Please login and specify the phrase to upload the video to our database!")
            #pred_txt.markdown(f'**Prediction:** {hyp[0]}')
            print("Can't receive frame (stream end?). Exiting ...")
            break

        end = time.time()

        frame_counter += 1
        fps_measurement = frame_counter/(end - start)
        bar.progress(frame_counter/num_frames)
        frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(cv2.resize(frm, (224, 224)))

        time.sleep(0.02)



main()
