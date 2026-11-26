package com.alejandro.similaritysearch

import com.google.gson.annotations.SerializedName
import java.util.concurrent.TimeUnit
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Query

const val BASE_URL = "http://10.0.2.2:8000/"

data class TextQuery(
    val collection: String,
    val text: String,
    val k: Int = 12
)

data class SearchResult(
    val id: Int,
    val label: String,
    val score: Double,
    val text: String?,
    @SerializedName("image_url") val imageUrl: String?
)

data class SearchResponse(
    val collection: String,
    val count: Int,
    val results: List<SearchResult>
)

interface SimilarityApi {
    @POST("search/text")
    suspend fun searchText(@Body query: TextQuery): SearchResponse

    @Multipart
    @POST("search/image")
    suspend fun searchImage(
        @Query("collection") collection: String,
        @Query("k") k: Int,
        @Part file: MultipartBody.Part
    ): SearchResponse
}

private val client = OkHttpClient.Builder()
    .readTimeout(60, TimeUnit.SECONDS)
    .writeTimeout(60, TimeUnit.SECONDS)
    .build()

val api: SimilarityApi = Retrofit.Builder()
    .baseUrl(BASE_URL)
    .client(client)
    .addConverterFactory(GsonConverterFactory.create())
    .build()
    .create(SimilarityApi::class.java)
