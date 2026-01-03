using System;
using System.Buffers;
using System.Collections.Generic;

namespace LlamaCpp
{
    public unsafe static class Util
    {

        /// <summary>
        /// Pins multiple Memory<T> instances and unpins them when this class is disposed.
        /// </summary>
        internal sealed class memory_pin_manager : IDisposable
        {
            private bool _disposed;
            // We store the MemoryHandle structs, which are disposable and control the pin state.
            private readonly List<MemoryHandle> _handles = new();

            // Finalizer is still useful, but primarily handles the unmanaged pinning reference
            ~memory_pin_manager()
            {
                Dispose();
            }

            /// <summary>
            /// Pins the provided Memory<T> instance and adds the handle to the manager.
            /// </summary>
            public unsafe IntPtr Pin<T>(Memory<T> memory)
                where T : unmanaged
            {
                if (_disposed)
                    throw new ObjectDisposedException("Cannot pin new Memory<T>, already disposed");

                // The key step: calling Pin() creates the MemoryHandle
                MemoryHandle handle = memory.Pin();
                _handles.Add(handle);

                // Return the fixed pointer (IntPtr) to the caller for use in P/Invoke
                return (IntPtr)handle.Pointer;
            }

            /// <inheritdoc />
            public void Dispose()
            {
                if (_disposed)
                    return;

                // Iterate through all handles and dispose of them, which unpins the memory.
                foreach (var handle in _handles)
                {
                    // MemoryHandle is a struct, so calling Dispose() unpins the buffer.
                    handle.Dispose();
                }
                _handles.Clear(); // Clear the list after disposal

                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
}
