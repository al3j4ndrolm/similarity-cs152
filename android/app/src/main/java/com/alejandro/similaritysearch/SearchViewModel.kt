package com.alejandro.similaritysearch

import android.content.ContentResolver
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class SearchViewModel : ViewModel() {
    var serverUrl by mutableStateOf(DEFAULT_URL)
    var collection by mutableStateOf("flowers")
    var query by mutableStateOf("")
    var queryImage by mutableStateOf<Uri?>(null)
    var results by mutableStateOf<List<SearchResult>>(emptyList())
    var loading by mutableStateOf(false)
    var error by mutableStateOf<String?>(null)

    fun changeCollection(name: String) {
        collection = name
        results = emptyList()
        queryImage = null
        error = null
    }

    fun searchText() {
        if (query.isBlank()) return
        queryImage = null
        doSearch { createApi(cleanUrl(serverUrl)).searchText(TextQuery(collection, query.trim())) }
    }

    fun searchImage(uri: Uri, resolver: ContentResolver) {
        queryImage = uri
        doSearch {
            val bytes = withContext(Dispatchers.IO) {
                resolver.openInputStream(uri)?.use { it.readBytes() } ?: ByteArray(0)
            }
            val body = bytes.toRequestBody("image/*".toMediaType())
            val part = MultipartBody.Part.createFormData("file", "photo.jpg", body)
            createApi(cleanUrl(serverUrl)).searchImage("flowers", 12, part)
        }
    }

    private fun doSearch(call: suspend () -> SearchResponse) {
        viewModelScope.launch {
            loading = true
            error = null
            try {
                results = call().results
            } catch (e: Exception) {
                results = emptyList()
                error = "Search failed: ${e.message}. Is the API running?"
            }
            loading = false
        }
    }
}
