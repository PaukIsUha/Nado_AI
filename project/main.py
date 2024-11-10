from flask import Flask, render_template, request, redirect, url_for
import os
from models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = Model()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'docx'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'ssts_file' not in request.files or 'hmi_file' not in request.files:
            return 'Файлы не найдены'
        ssts_file = request.files['ssts_file']
        hmi_file = request.files['hmi_file']
        if ssts_file.filename == '' or hmi_file.filename == '':
            return 'Файлы не выбраны'
        if ssts_file and allowed_file(ssts_file.filename) and hmi_file and allowed_file(hmi_file.filename):
            ssts_filename = 'ssts.docx'
            hmi_filename = 'hmi.docx'
            ssts_file.save(os.path.join(app.config['UPLOAD_FOLDER'], ssts_filename))
            hmi_file.save(os.path.join(app.config['UPLOAD_FOLDER'], hmi_filename))
            return redirect(url_for('inference'))
    return render_template('index.html')


@app.route('/inference')
def inference():
    res = model.fit("uploads/ssts.docx", "uploads/hmi.docx")

    difference = res.difference
    description = res.description
    compliance_level = res.compl

    return render_template('result.html', difference=difference, description=description, compliance_level=compliance_level)


@app.route('/feedback', methods=['POST'])
def feedback():
    rating = request.form.get('rating')
    comment = request.form.get('comment')

    print(f"Оценка пользователя: {rating}")
    print(f"Комментарий пользователя: {comment}")

    return render_template('thank_you.html')


if __name__ == '__main__':
    app.run(debug=True)
