package com.jakereyes.mdds;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.jakereyes.mdds.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button picture;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        Button selectImageButton = findViewById(R.id.button1);

        picture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                } else {
                    openCamera();
                }
            }
        });

        selectImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, 2);
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, 1);
    }

    @SuppressLint("SetTextI18n")
    public void classifyImage(Bitmap image) {
        Model model = null;
        try {
            model = Model.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[224 * 224];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < 224; i++) {
                for (int j = 0; j < 224; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Anthracnose", "Healthy", "Stem end Rot", "Not a Mango", "Scab"};
            String resultText = classes[maxPos];
            result.setText(resultText);

// Display specific information based on the classification result
            switch (resultText) {
                case "Anthracnose":
                    result.setText("Anthracnose\n(Colletotrichum gloeosporioides)\n\n" +
                            "The disease is more common on young fruits and during transit and storage. Latent infection during pre-harvest stage is responsible for post harvest rots. On storage, black spots are produced. Initially, the spots are round but later form large irregular blotches on the entire fruits. The spots have large deep cracks and the fungus penetrates deep into the fruit causing extensive rotting.\n\n" +
                            "Control:\n" +
                            "i. Pre-harvest infections can be managed by spraying copper-based fungicides after completion of heavy showers.\n" +
                            "ii. Post-harvest infections can be managed as pre-harvest sprays in the field to reduce the latent infection and treatment of the fruit with hot water/fungicides after harvest to eradicate leftover latent infection.");
                    break;
                case "Stem end Rot":
                    result.setText("Stem End Rot\n( Lasiodiplodia theobromae, Phomospsis mangiferae, Dothiorella dominicana)\n\n" +
                            "The fruit, while ripening, suddenly becomes brown to black typically at the stem end. Within two to three days, the whole fruit becomes black and the disease progresses downwards, thus involving half of the area of the fruits. Though the flush of the whole fruit often wrinkles are also observed. Affected skin remains firm but decay sets into the pulp below and emits an unpleasant odor.\n\n" +
                            "Control:\n" +
                            "i. Prompt and proper handling of the fruit can minimize disease incidence.\n" +
                            "ii. Fruit should be harvested with a 10mm stalk.\n" +
                            "iii. Pre-harvest sprays of any systemic fungicides or copper-based fungicides reduce the incidence of SER.");
                    break;
                case "Scab":
                    result.setText("Scab\n(Elsinoe mangiferae)\n\n" +
                            "Scab is a fungal disease affecting mango fruits. It manifests as raised, corky scabs or spots on the fruit surface, often surrounded by a yellow halo. Severe infections can cause distortion and cracking of the fruit. Scab can lead to reduced fruit quality and yield if left unmanaged.\n\n" +
                            "Control:\n" +
                            "i. Cultural practices such as pruning to improve airflow and reduce humidity can help prevent scab.\n" +
                            "ii. Application of fungicides, especially during periods of high humidity, can effectively manage scab.\n" +
                            "iii. Removal and destruction of infected plant debris can reduce the inoculum source for future infections.");
                    break;
            }



            // Update the bar chart with the latest data

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            closeModel(model);
        }
    }


    private void closeModel(Model model) {
        if (model != null) {
            model.close();
        }
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Bundle extras = data.getExtras();
            if (extras != null) {
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                if (imageBitmap != null) {
                    handleImage(imageBitmap);
                }
            }
        } else if (requestCode == 2 && resultCode == RESULT_OK && data != null) {
            try {
                Bitmap imageBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(data.getData()));
                if (imageBitmap != null) {
                    handleImage(imageBitmap);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void handleImage(Bitmap imageBitmap) {
        int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
        imageBitmap = ThumbnailUtils.extractThumbnail(imageBitmap, dimension, dimension);
        imageView.setImageBitmap(imageBitmap);

        Bitmap scaledImage = Bitmap.createScaledBitmap(imageBitmap, 224, 224, false);
        classifyImage(scaledImage);
    }
}
