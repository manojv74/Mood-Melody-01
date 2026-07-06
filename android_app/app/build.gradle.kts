plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.compose.compiler)
}

android {
    namespace = "com.example.emotiondetection"
    compileSdk = 36
    
    defaultConfig {
        applicationId = "com.example.emotiondetection"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    /*signingConfigs {
        // For CI/CD or environment-based signing (e.g., GitHub Actions), use this block:
        create("release") {
            storeFile = file(System.getenv("ORG_GRADLE_PROJECT_storeFile"))
            storePassword = System.getenv("ORG_GRADLE_PROJECT_storePassword")
            keyAlias = System.getenv("ORG_GRADLE_PROJECT_keyAlias")
            keyPassword = System.getenv("ORG_GRADLE_PROJECT_keyPassword")
        }
        // For local Android Studio builds, you can comment out the above block entirely.
        // Android Studio's "Generate Signed Bundle / APK" wizard lets you pick the keystore and credentials interactively.
    }*/

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            isDebuggable = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            //signingConfig = signingConfigs.getByName("release") //comment this line if you want to use Android Studio's interactive signing wizard
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a")
            isUniversalApk = false
        }
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        mlModelBinding = true
    }
    packaging {
        resources {
            excludes += listOf("META-INF/LICENSE.md", "META-INF/LICENSE-notice.md")
        }
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)    
    androidTestImplementation(libs.androidx.espresso.core)
    
    // LiteRT core dependencies (replaces TensorFlow Lite)
    implementation(libs.litert)
    
    // LiteRT GPU acceleration dependencies
    implementation(libs.litert.gpu)
    
    // Compose dependencies
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.graphics)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.activity)   
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    
    // Compose debug tools
    debugImplementation(libs.compose.ui.tooling)
    debugImplementation(libs.compose.ui.test.manifest)    
    
    // ML Kit dependencies
    implementation(libs.face.detection)
    implementation(libs.vision.common)

    // CameraX for continuous monitoring
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)

    // Networking: simple HTTP client used to request generated tracks from the server
    implementation("com.squareup.okhttp3:okhttp:4.11.0")

    // Coroutines (if you want to use lifecycleScope or other coroutine helpers)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
