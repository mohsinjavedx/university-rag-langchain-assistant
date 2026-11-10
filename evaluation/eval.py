import sys
import math
from xmlrpc import client
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from openai import OpenAI
from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context