import cv2
import streamlit as st
from sqlalchemy import text
from preprocess import read_image,extract_id_card,save_image
from ocr_engines import extract_text
from postprocess import extract_information
from face_verification import detect_and_extract_face,face_comparison,get_face_embeddings
from mysql_dboperation import fetch_records,insert_records,check_duiplicates
