import glob
import uuid
import yaml
from flask import Flask, request, flash, redirect, send_from_directory, render_template
from imageai.Detection import ObjectDetection
from flask_celery import make_celery
import os

config_path = "config.yml"
config = yaml.load(open(config_path))

app = Flask(__name__)
app.config.update(config)

celery = make_celery(app)


@celery.task(name='image.processing')
def retush(filename):
    exec_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(exec_path, "resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel(detection_speed="flash")

    listis = detector.detectObjectsFromImage(
        input_image=os.path.join(app.config['UPLOAD_FOLDER'], filename),
        output_image_path=os.path.join(app.config['RESULT_FOLDER'], filename)
    )
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/main', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename, file_extension = os.path.splitext(file.filename)
            filename = str(uuid.uuid4()) + file_extension
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            retush.apply_async(args=[filename], countdown=1)
            return redirect('/process/' + filename)
        else:
            flash('Неверный формат или файл отсуствует')
            return render_template('main.html')
    elif request.method == 'GET':
        return render_template('main.html')


@app.route('/')
def start():
    ## Очистка папки results, используется временно
    if len(glob.glob('results/*')) > 3:
        files = glob.glob('results/*')
        for f in files:
            os.remove(f)
    ##
    return render_template('index.html')


@app.route('/process/<filename>', methods=['GET'])
def task_processing(filename):
    if request.method == 'GET':
        return render_template('temp.html', filename=filename)


@app.route('/result/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/examples')
def examples():
    return render_template('examples.html')


@app.route('/faq')
def faq():
    return render_template('faq.html')


if __name__ == "__main__":
    app.run(debug=False)

''' 
В терминал: celery -A app.celery worker --loglevel=info
Для запуска нескольких worker-ов
celery -A app.celery worker -l info --concurrency=2 -n worker1@hostname
celery -A app.celery worker -l info --concurrency=2 -n worker2@hostname
'''
