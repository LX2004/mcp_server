# load text from file
with open('sequence.txt', 'r') as f:
  sequence = f.read()

# post with text form data 
import requests

url = 'https://weblogo.berkeley.edu/logo.cgi'
data = {
  'sequence': sequence,
  'format': 'PNG',
  'logowidth': 18,
  'logoheight': 5,
  'logounits': 'cm',
  'command': 'Create Logo',
  'kind': 'AUTO',
  'firstnum': 1,
  'logostart': '',
  'logoend': '',
  'smallsamplecorrection': 'on',
  'symbolsperline': 32,
  'res': 96,
  'res_units': 'ppi',
  'antialias': 'on',
  'title': '',
  'barbits': '',
  'yaxis': 'on',
  'yaxis_label': 'bits',
  'xaxis': 'on',
  'xaxis_label': '',
  'showends': 'on',
  'shrink': 0.5,
  'fineprint': 'on',
  'ticbits': 1,
  'colorscheme': 'DEFAULT',
  'symbol1': 'KRH',
  'color1': 'green',
  'rgb1': '',
  'symbol2': 'DE',
  'color2': 'blue',
  'rgb2': '',
  'symbol3': 'AVLIPWFM',
  'color3': 'red',
  'rgb3': '',
  'symbol4': '',
  'color4': 'black',
  'rgb4': '',
  'symbol5': '',
  'color5': 'purple',
  'rgb5': '',
  'symbol6': '',
  'color6': 'orange',
  'rgb6': '',
  'symbol7': '',
  'color7': 'black',
  'rgb7': '',
  'color0': 'black',
  'rgb0': '',
}

response = requests.post(url, data=data)

# save as image
with open('image.png', 'wb') as f:
  f.write(response.content)
