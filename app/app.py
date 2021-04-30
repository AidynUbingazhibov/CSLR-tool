import streamlit as st
import pandas as pd
import streamlit as st
import numpy
import sys
import os
import tempfile
sys.path.append(os.getcwd())
from app_model import main as m
from app_model import trigger_rerun
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


# Security

#passlib,hashlib,bcrypt,scrypt

import hashlib

def make_hashes(password):

	return hashlib.sha256(str.encode(password)).hexdigest()



def check_hashes(password,hashed_text):

	if make_hashes(password) == hashed_text:

		return hashed_text

	return False

# DB Management

import sqlite3

conn = sqlite3.connect('data.db')

c = conn.cursor()

# DB  Functions

def create_usertable():

	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')





def add_userdata(username,password):

	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))

	conn.commit()



def login_user(username,password):

	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))

	data = c.fetchall()

	return data





def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data


def main():
	#st.title("")
	menu = ["Home", "Login","SignUp"]
	choice = st.sidebar.selectbox("Menu",menu)
	result = 0

	if choice == "Login":
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')

		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))

			if result:
				st.success("Logged In as {}".format(username))
				with open("/app/app/upload.txt", "w") as f:
					f.write("1")

			else:
				st.warning("Incorrect Username/Password")



	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")
	elif choice == "Home":
		with open("/app/app/upload.txt", "r") as f:
			res = int(f.readline())

		if res:
			st.sidebar.markdown("Signer id: **51**")
		m()


if __name__ == '__main__':
	main()
