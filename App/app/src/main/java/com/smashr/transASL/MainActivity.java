package com.smashr.transASL;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.os.Process;
import android.view.View;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "SignTranslate";
    private static final int REQUEST_FOR_CAMERA = 1;
    private final int startRecognize = R.drawable.ic_baseline_camera_24;
    private final int stopRecognize = R.drawable.ic_baseline_cancel_24;
    private ImageAnalysis handAnalysis = null;
    private ImageButton mButton = null;
    private ProcessCameraProvider cameraProvider = null;
    private PreviewView previewView = null;
    protected SignAnalyzer mSignAnalyzer = null;
    private TextView prediction;

    protected Handler mAnalyserHandler = new Handler(Looper.getMainLooper()){
        @Override
        public void handleMessage(Message msg){
            if (msg.what == SignAnalyzer.PREDICTION) {
                prediction.setText((String)msg.obj);
            } else {
                super.handleMessage(msg);
            }
        }
    };

    private ByteBuffer loadFilePath() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mobnet_kaggle_192.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private ActivityCompat.OnRequestPermissionsResultCallback requestPermissionsResultCallback = new ActivityCompat.OnRequestPermissionsResultCallback() {
        @Override
        public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
            if(requestCode == REQUEST_FOR_CAMERA){
                if(permissionGranted()) {
                    try {
                        startPreview();
                    } catch (ExecutionException | InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                else
                    Toast.makeText(MainActivity.this, "App can't function without Camera permissions", Toast.LENGTH_SHORT).show();
            }
        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prediction = findViewById(R.id.predicted_sign);
        previewView = findViewById(R.id.camera_surface);
        mButton = findViewById(R.id.toggle_button);
        mButton.setOnClickListener(view -> toggle());
        mButton.setTag(startRecognize);
        if(permissionGranted()){
            try {
                startPreview();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }
        else{
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.CAMERA}, REQUEST_FOR_CAMERA);
        }
        createAnalyser();
    }

    private void createAnalyser(){
        final int TASK_COMPLETE = 3001;
        Handler mCreationHandler = new Handler(Looper.getMainLooper()){
            @Override
            public void handleMessage(Message msg){
                if (msg.what == TASK_COMPLETE) {
                    mSignAnalyzer = (SignAnalyzer) msg.obj;
                } else {
                    super.handleMessage(msg);
                }
            }
        };

        Thread createAnalyser = new Thread(() -> {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
            try {
                SignAnalyzer signAnalyzer = new SignAnalyzer(MainActivity.this,new Interpreter(loadFilePath()),mAnalyserHandler);
                Message toMainAct = mCreationHandler.obtainMessage(TASK_COMPLETE,signAnalyzer);
                toMainAct.sendToTarget();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        createAnalyser.start();
    }


    private boolean permissionGranted(){
        return (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED);
    }

    public void startPreview() throws ExecutionException, InterruptedException {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProvider = cameraProviderFuture.get();
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.createSurfaceProvider());
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(MainActivity.this,cameraSelector,preview);
    }
    public void startAnalyser(){
        Executor cameraExecutor = Executors.newSingleThreadExecutor();
        handAnalysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        handAnalysis.setAnalyzer(cameraExecutor,mSignAnalyzer);
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.createSurfaceProvider());
        CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this,cameraSelector,preview,handAnalysis);
    }


    public void toggle() {
        int drawableId = (Integer) mButton.getTag();
        if(drawableId == startRecognize){
            //TODO:
            if(mSignAnalyzer == null){
                Toast.makeText(this, "Loading Recognition model", Toast.LENGTH_SHORT).show();
            }
            else {
                startAnalyser();
                prediction.setVisibility(View.VISIBLE);
                mButton.setTag(stopRecognize);
                mButton.setImageResource(stopRecognize);
            }
        }
        else {
            //TODO:
            cameraProvider.unbind(handAnalysis);
            prediction.setVisibility(View.INVISIBLE);
            mButton.setTag(startRecognize);
            mButton.setImageResource(startRecognize);
        }
    }
}
