package com.smashr.transASL;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

class SignAnalyzer implements ImageAnalysis.Analyzer {

    private static String TAG = "SignAnalyzer";
    private static final int NUM_OUTPUTS = 29;
    private Activity mainActivity;
    private Handler mHandler;
    public static final int PREDICTION = 5005;
    private static final int CHANNEL_SIZE = 3;
    private static final int IMAGE_HEIGHT = 192;
    private static final int IMAGE_WIDTH = 192;
    private static final int bufferSize = NUM_OUTPUTS* Float.SIZE/Byte.SIZE;
    private final int modelInputSize = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNEL_SIZE * Float.SIZE/Byte.SIZE;
    private final float IMAGE_MEAN = 0.0f;
    private final float IMAGE_STD = 255.0f;
    private Interpreter tfliteModel;
    private ArrayList<String> outputLabels = new ArrayList<>();

//    class Pair{
//        float num;
//        int idx;
//    }
//    private String getTop3Prob(FloatBuffer buffer){
//        Pair[] array = new Pair[buffer.array().length];
//        for(int i=0;i<NUM_OUTPUTS;++i){
//            array[i].num = buffer.get(i);
//            array[i].idx = i;
//        }
//        Arrays.sort(array, new Comparator<Pair>() {
//            @Override
//            public int compare(Pair pair, Pair t1) {
//                return Float.compare(t1.num, pair.num);
//            }
//        });
//        return new String(outputLabels.get(0)+"  "+outputLabels.get(1)+"  "+outputLabels.get(2));
//    }
    SignAnalyzer(Activity activity, Interpreter tflite, Handler handler) throws IOException {
        mainActivity = activity;
        tfliteModel = tflite;
        mHandler = handler;

        initLabels();
    }

    private void initLabels() throws IOException {
        InputStream inputStream = mainActivity.getAssets().open("labels.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String label;
        while((label = reader.readLine()) != null){
            outputLabels.add(label);
        }
    }

    @Override
    public void analyze(@NonNull ImageProxy image){
        Bitmap resizedImage = Bitmap.createScaledBitmap(toBitmap(image),IMAGE_WIDTH,IMAGE_HEIGHT,true);
        ByteBuffer modelInput = convertBitmapToByteBuffer(resizedImage);
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());

        tfliteModel.run(modelInput,modelOutput);
        modelOutput.rewind();
        FloatBuffer probabilities = modelOutput.asFloatBuffer();
        int idx = 0;
        float maxProb = 0.0f;
        int id = 0;
        for (String label: outputLabels) {
            float prob = probabilities.get(idx++);
            if(maxProb < prob){
                maxProb = prob;
                id = idx-1;
            }
            Log.i(TAG, String.format("%s: %1.4f", label, prob));
        }
        String predicted = "nothing";
        if(maxProb*100 > 76.0f)
            predicted = outputLabels.get(id);
        Message toBeSent = mHandler.obtainMessage(PREDICTION,predicted);
        toBeSent.sendToTarget();
        image.close();
    }
    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        Bitmap bInput = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        Matrix matrix = new Matrix();
        matrix.preScale(-1.0f, 1.0f);
        return Bitmap.createBitmap(bInput, 0, 0, bInput.getWidth(), bInput.getHeight(), matrix, true);
    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap){
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(modelInputSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[IMAGE_WIDTH * IMAGE_HEIGHT];
        bitmap.getPixels(pixels,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        for(int pixelVal: pixels){
        	byteBuffer.putFloat((((pixelVal>>16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            byteBuffer.putFloat((((pixelVal>>8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            byteBuffer.putFloat((((pixelVal) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        }
        bitmap.recycle();
        return byteBuffer;
    }


}

