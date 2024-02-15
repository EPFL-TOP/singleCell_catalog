# bokeh_app.py
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Rect, Button, Slider
from bokeh.events import SelectionGeometry
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.layouts import column, row
import numpy as np
import sys
sys.path.append('/Users/helsens/Software/github/EPFL-TOP/singleCell_catalog')

from segmentation.models import ROI

class BokehApp:
    def __init__(self, image_data, file):
        self.num_images, self.height, self.width = image_data.shape
        self.image_data = image_data
        self.image_index = 0
        self.selected_region = {'x': [], 'y': []}

        # Retrieve or create ImageModel instance
        self.image_instance, _ = ROI.objects.get_or_create(sample=file)

        self.source = ColumnDataSource(data=self.selected_region)
        self.p = figure(title="Select Region")
        self.p.image(image=[image_data[self.image_index]], x=0, y=0, dw=self.width, dh=self.height)
        self.rect = Rect(x='x', y='y', width='width', height='height', fill_alpha=0.1, line_color='red')
        self.p.add_glyph(self.source, self.rect)

        self.play_button = Button(label="Play", button_type="success")
        self.slider = Slider(start=0, end=self.num_images-1, value=0, step=1, title="Image Index")

        self.play_button.on_click(self.play_callback)
        self.slider.on_change('value', self.slider_callback)

        self.layout = column(row(self.play_button, self.slider), self.p)

        # Configure CORS settings
        self.allow_websocket_origin = ["localhost:8001"]  # Replace with actual origin
        self.allow_origin = ["localhost:8001"]  # Replace with actual origin

    def play_callback(self):
        if self.play_button.label == "Play":
            self.play_button.label = "Pause"
            self.timer = curdoc().add_periodic_callback(self.update_image_index, 1000)
        else:
            self.play_button.label = "Play"
            curdoc().remove_periodic_callback(self.timer)

    def update_image_index(self):
        if self.image_index < self.num_images - 1:
            self.image_index += 1
        else:
            self.image_index = 0
        self.p.image_url(url=[self.image_data[self.image_index]], x=0, y=0, dw=self.width, dh=self.height)

    def slider_callback(self, attr, old, new):
        self.image_index = new
        self.p.image_url(url=[self.image_data[self.image_index]], x=0, y=0, dw=self.width, dh=self.height)

    def handle_selection_change(self, event):
        if isinstance(event, SelectionGeometry):
            selected_data = self.source.data
            selected_data['x'] = [event.geometry['x0'], event.geometry['x0'], event.geometry['x1'], event.geometry['x1']]
            selected_data['y'] = [event.geometry['y0'], event.geometry['y1'], event.geometry['y1'], event.geometry['y0']]
            selected_data['width'] = [event.geometry['x1'] - event.geometry['x0']]
            selected_data['height'] = [event.geometry['y1'] - event.geometry['y0']]
            self.source.data = selected_data
            
            # Update ImageModel
            #self.image_instance.selected_region = {'x': selected_data['x'], 'y': selected_data['y']}
            self.image_instance.min_col = event.geometry['x0']
            self.image_instance.max_col = event.geometry['x1']
            self.image_instance.min_row = event.geometry['y0']
            self.image_instance.max_row = event.geometry['y1']
            self.image_instance.save()
            print("Model updated successfully")

def create_bokeh_app(image_data):
    return Application(FunctionHandler(lambda doc: BokehApp(image_data).layout))

# Create and start the Bokeh server with CORS settings
#server = Server({'/bokeh_app': create_bokeh_app}, allow_websocket_origin=["localhost:8001"], allow_origin=["localhost:8001"])
#server.start()