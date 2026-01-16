using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace LlamaCpp
{
    public static class NativeLibraryPath
    {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern IntPtr GetModuleHandle(string lpModuleName);

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern uint GetModuleFileName(IntPtr hModule, StringBuilder lpFilename, int nSize);

        public static string GetDllPath(string dllName)
        {
            IntPtr hModule = GetModuleHandle(dllName);
            if (hModule == IntPtr.Zero)
                return null;

            var sb = new StringBuilder(1024);
            GetModuleFileName(hModule, sb, sb.Capacity);

            return Path.GetDirectoryName(sb.ToString());
        }
#elif UNITY_ANDROID && !UNITY_EDITOR
        public static string GetAndroidNativeLibraryPath() 
        {
            using (var player = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            {
                var activity = player.GetStatic<AndroidJavaObject>("currentActivity");
                var info = activity.Call<AndroidJavaObject>("getApplicationInfo");
                return info.Get<string>("nativeLibraryDir");
            }
        }
#endif
    }

    public static class Backend
    {
        static int count = 0;

        public static void EnableLog()
        {
            Native.llama_log_set(NativeLogCallback, IntPtr.Zero);
        }

        public static void Init()
        {
            if (count == 0)
            {
                Native.llama_max_devices();

#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
                string packagePath = NativeLibraryPath.GetDllPath("llama.dll");
                Native.ggml_backend_load_all_from_path(packagePath);
#elif UNITY_ANDROID && !UNITY_EDITOR
                string packagePath = NativeLibraryPath.GetAndroidNativeLibraryPath();
                Native.ggml_backend_load_all_from_path(packagePath);
#else
                Native.ggml_backend_load_all();
#endif

                Native.llama_backend_init();
            }
            count++;
        }

        public static void Free()
        {
            count--;
            if (count == 0)
            {
                Native.llama_backend_free();
                Native.llama_log_set(null, IntPtr.Zero);
            }
        }

        [AOT.MonoPInvokeCallback(typeof(Native.ggml_log_callback))]
        static void NativeLogCallback(int level, string text, IntPtr user_data)
        {
            if (level >= 4)
            {
                Debug.LogWarning("[Llama Native] " + text);
            }
            else
            {
                Debug.Log("[Llama Native] " + text);
            }
        }

    }
}
