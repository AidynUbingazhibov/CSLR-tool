FROM nvcr.io/nvidia/pytorch:20.12-py3

# Install linux packages
RUN echo "nameserver 10.1.1.50" | tee /etc/resolv.conf > /dev/null
RUN apt-get update ##[edited]
RUN apt-get install -y screen libgl1-mesa-glx
# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install gsutil

ENV APP_HOME /app/
WORKDIR ${APP_HOME}


COPY requirements.txt . 
RUN pip3 install -r requirements.txt

COPY . .
RUN wget https://pjreddie.com/media/files/yolov3.weights
RUN mv yolov3.weights data/

# prepare building of component 
WORKDIR ${APP_HOME}/components/custom_slider/frontend/
#RUN npm install
#RUN npm run build

EXPOSE 8501

WORKDIR ${APP_HOME}

CMD sh setup.sh && streamlit run app/streamlit-app.py












