"""
Django settings for singleCell_catalog project.

Generated by 'django-admin startproject' using Django 3.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/3.2/ref/settings/
"""
TEST=False
from pathlib import Path
import os#, sys
#from bokeh.settings import bokehjs_path, settings as bokeh_settings
from bokeh.settings import bokehjsdir, settings as bokeh_settings
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
#VMachine
#if os.path.isdir('/home/helsens/Software/segmentationTools/cellgmenter/main'):
#    sys.path.append('/home/helsens/Software/UPOATES_catalog')

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/3.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-rivui%0@fa(act8yw4ou9jvh6qf@7qhmntaaq+squb2k(05912'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['svvm0003.xaas.epfl.ch:8001','127.0.0.1','localhost', 'localhost:8001', '0.0.0.0:8001', 'sv-upoates.epfl.ch']

GRAPH_MODELS = {
  'all_applications': True,
  'group_models': True,
  'app_labels': ["singleCell_catalog"],
  }


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_extensions',
    'segmentation.apps.SegmentationConfig', 
    'corsheaders',
    'channels',
    'bokeh_django',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',  # Add this line
    'django_cprofile_middleware.middleware.ProfilerMiddleware',
]




ROOT_URLCONF = 'singleCell_catalog.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

ASGI_APPLICATION = 'singleCell_catalog.asgi.application'
#WSGI_APPLICATION = 'singleCell_catalog.wsgi.application'

DATA_UPLOAD_MAX_NUMBER_FIELDS = 10240 # higher than the count of fields

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases

DATABASES = None
import os
import accesskeys as accessk

if os.path.isdir("/Users/helsens/Software/github/EPFL-TOP/") or TEST==True:
    #TEST LOCAL
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
elif os.path.isdir("/home/helsens/Software/") and TEST==False:
    #PROD
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': accessk.DB_name,
            'USER': accessk.DB_user,
            'PASSWORD': accessk.DB_password,
            'HOST': '127.0.0.1',
            'PORT': '3306',
        }
    }


elif os.path.isdir(r"C:\Users\helsens\software"):
    #PROD
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': accessk.DB_name,
            'USER': accessk.DB_user,
            'PASSWORD': accessk.DB_password,
            'HOST': '127.0.0.1',
#            'PORT': '3305',#LOCAL HIVE PORT
            'PORT': '3336', #SV VM Port
        }
    }

# Password validation
# https://docs.djangoproject.com/en/3.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/3.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Redirect to home URL after login (Default redirects to /accounts/profile/)
LOGIN_REDIRECT_URL = '/segmentation'


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/3.2/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


STATICFILES_DIRS = [
   os.path.join(BASE_DIR, "static"),
   #bokehjs_path()
   bokehjsdir()
]

STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
    'bokeh_django.static.BokehExtensionFinder',
)

bokeh_settings.resources = 'server'



CORS_ALLOWED_ORIGINS = [
    "http://0.0.0.0:8001",  # Add the origin of your Django server
]

CORS_ALLOW_METHODS = [
    "GET",
    "POST",
    "OPTIONS"
]
