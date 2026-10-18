/* Calls the C API exactly as an external C program would (this file is compiled as C99 with only <dooms/dooms_c.h>): version/error plumbing, argument validation and
 * the failure of every open call on missing files. The real-model behaviour is covered from Python (python/tests), except two parts that also run here through
 * C on real models: the asynchronous VLA runner when DOOMS_VLA_DIR names the converted SmolVLA files (CTest c_api_smolvla), and the LLM's token counts and
 * finish reasons when DOOMS_LLM_FILE names a chat-tuned GGUF (CTest c_api_llm). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dooms/dooms_c.h"

static int failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            fprintf(stderr, "  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);   \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

int main(void) {
    CHECK(dooms_c_api_version() == DOOMS_C_API_VERSION);
    CHECK(strlen(dooms_version()) >= 5);
    CHECK(strcmp(dooms_code_name(DOOMS_OK), "Ok") == 0);
    CHECK(strcmp(dooms_code_name(DOOMS_NOT_FOUND), "NotFound") == 0);
    CHECK(strlen(dooms_last_error()) == 0);

    /* closing NULL is a no-op; calls on a NULL handle fail cleanly */
    dooms_vlm_close(NULL);
    dooms_vla_close(NULL);
    dooms_image img = {2, 2, (const uint8_t *) "\0\0\0\0\0\0\0\0\0\0\0\0"};
    /* vlm */
    dooms_vlm * vlm = (dooms_vlm *) 1;
    dooms_vlm_paths vp;
    memset(&vp, 0, sizeof vp);
    CHECK(dooms_vlm_open(&vp, NULL, &vlm) == DOOMS_INVALID_ARGUMENT);
    CHECK(vlm == NULL);
    vp.struct_size = sizeof vp;
    vp.vision = "/nonexistent/v.mlpackage";
    vp.text = "/nonexistent/t.gguf";
    vp.config = "/nonexistent/c.gguf";
    {
        dooms_code c = dooms_vlm_open(&vp, NULL, &vlm);
        CHECK(c == DOOMS_NOT_FOUND || c == DOOMS_UNSUPPORTED); /* UNSUPPORTED in a build without llama.cpp */
    }
    char text[16];
    size_t needed = 0;
    CHECK(dooms_vlm_generate(NULL, &img, "hi", 8, text, sizeof text, &needed) == DOOMS_INVALID_ARGUMENT);

    CHECK(dooms_vlm_generate_stream(NULL, &img, "hi", 8, NULL, NULL) == DOOMS_INVALID_ARGUMENT);

    /* llm */
    dooms_llm * llm = (dooms_llm *) 1;
    CHECK(dooms_llm_open(NULL, NULL, &llm) == DOOMS_INVALID_ARGUMENT);
    CHECK(llm == NULL);
    CHECK(dooms_llm_open("/nonexistent/m.gguf", NULL, NULL) == DOOMS_INVALID_ARGUMENT);
    {
        dooms_code c = dooms_llm_open("/nonexistent/m.gguf", NULL, &llm);
        CHECK(c == DOOMS_NOT_FOUND || c == DOOMS_UNSUPPORTED);
        dooms_llm_options lo;
        memset(&lo, 0, sizeof lo);
        CHECK(dooms_llm_open("/nonexistent/m.gguf", &lo, &llm) == DOOMS_INVALID_ARGUMENT); /* struct_size 0 */
    }
    dooms_llm_close(NULL);
    CHECK(dooms_llm_context_size(NULL) == 0);
    CHECK(dooms_llm_generate(NULL, "hi", 8, NULL, NULL, 0, text, sizeof text, &needed) == DOOMS_INVALID_ARGUMENT);
    CHECK(dooms_llm_generate_stream(NULL, "hi", 8, NULL, NULL, 0, NULL, NULL) == DOOMS_INVALID_ARGUMENT);
    CHECK(dooms_llm_chat(NULL, NULL, 0, 8, NULL, NULL, 0, text, sizeof text, &needed) == DOOMS_INVALID_ARGUMENT);
    CHECK(dooms_llm_chat_stream(NULL, NULL, 0, 8, NULL, NULL, 0, NULL, NULL) == DOOMS_INVALID_ARGUMENT);

    {
        dooms_llm_generation_info gi;
        int32_t count = 5;
        memset(&gi, 0, sizeof gi);
        CHECK(dooms_llm_last_generation(NULL, &gi) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_llm_count_tokens(NULL, "hi", 1, &count) == DOOMS_INVALID_ARGUMENT);
        CHECK(count == 0);
        CHECK(dooms_llm_count_tokens(NULL, "hi", 1, NULL) == DOOMS_INVALID_ARGUMENT);
        count = 5;
        CHECK(dooms_llm_count_chat_tokens(NULL, NULL, 0, &count) == DOOMS_INVALID_ARGUMENT);
        CHECK(count == 0);
    }
    if (getenv("DOOMS_LLM_FILE") && *getenv("DOOMS_LLM_FILE")) {
        dooms_llm_options lo;
        memset(&lo, 0, sizeof lo);
        lo.struct_size = sizeof lo;
        lo.n_ctx = 512;
        llm = NULL;
        CHECK(dooms_llm_open(getenv("DOOMS_LLM_FILE"), &lo, &llm) == DOOMS_OK);
        if (llm) {
            dooms_llm_generation_info gi;
            int32_t count = 0;
            char answer[1024];
            memset(&gi, 0, sizeof gi);
            CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_INVALID_ARGUMENT); /* struct_size unset */
            gi.struct_size = sizeof gi;
            CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_OK);
            CHECK(gi.finish_reason == DOOMS_FINISH_NONE && gi.prompt_tokens == 0 && gi.completion_tokens == 0);
            CHECK(dooms_llm_count_tokens(llm, "", 1, &count) == DOOMS_INVALID_ARGUMENT);
            CHECK(dooms_llm_count_tokens(llm, NULL, 1, &count) == DOOMS_INVALID_ARGUMENT);
            CHECK(dooms_llm_count_tokens(llm, "Count from one to ten: one, two, three,", 1, &count) == DOOMS_OK);
            CHECK(count > 5);
            /* a token limit that is surely hit: LENGTH, exactly that many tokens, the prompt as counted */
            CHECK(dooms_llm_generate(llm, "Count from one to ten: one, two, three,", 3, NULL, NULL, 0, answer, sizeof answer, NULL) == DOOMS_OK);
            CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_OK);
            CHECK(gi.finish_reason == DOOMS_FINISH_LENGTH);
            CHECK(gi.completion_tokens == 3);
            CHECK(gi.prompt_tokens == count);
            /* a stop string that matches: STOP before the limit */
            {
                const char * stops[1] = {"five"};
                CHECK(dooms_llm_generate(llm, "Count from one to ten: one, two, three,", 40, NULL, stops, 1, answer, sizeof answer, NULL) == DOOMS_OK);
                CHECK(strstr(answer, "five") == NULL);
                CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_OK);
                CHECK(gi.finish_reason == DOOMS_FINISH_STOP);
                CHECK(gi.completion_tokens > 0 && gi.completion_tokens < 40);
            }
            /* a chat that ends by itself: STOP, and the prompt is what count_chat_tokens says */
            {
                dooms_chat_message m[1] = {{"user", "Reply with the single word: pong"}};
                int32_t chat_count = 0;
                CHECK(dooms_llm_count_chat_tokens(llm, m, 1, &chat_count) == DOOMS_OK);
                CHECK(dooms_llm_count_chat_tokens(llm, m, 0, &chat_count) == DOOMS_INVALID_ARGUMENT);
                CHECK(dooms_llm_count_chat_tokens(llm, m, 1, &chat_count) == DOOMS_OK);
                CHECK(dooms_llm_chat(llm, m, 1, 64, NULL, NULL, 0, answer, sizeof answer, NULL) == DOOMS_OK);
                CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_OK);
                CHECK(gi.finish_reason == DOOMS_FINISH_STOP);
                CHECK(gi.prompt_tokens == chat_count);
                CHECK(gi.completion_tokens >= 1 && gi.completion_tokens < 64);
            }
            /* a failed call resets the record */
            CHECK(dooms_llm_generate(llm, "", 3, NULL, NULL, 0, answer, sizeof answer, NULL) == DOOMS_INVALID_ARGUMENT);
            CHECK(dooms_llm_last_generation(llm, &gi) == DOOMS_OK);
            CHECK(gi.finish_reason == DOOMS_FINISH_NONE && gi.completion_tokens == 0);
            /* an older caller with a smaller struct only gets the fields it has, and its struct_size is kept */
            {
                struct { size_t struct_size; int32_t prompt_tokens; int32_t completion_tokens; int32_t sentinel; } old_info;
                CHECK(dooms_llm_generate(llm, "Count from one to ten: one, two, three,", 2, NULL, NULL, 0, answer, sizeof answer, NULL) == DOOMS_OK);
                memset(&old_info, 0, sizeof old_info);
                old_info.sentinel = 77;
                old_info.struct_size = offsetof(dooms_llm_generation_info, finish_reason);
                CHECK(dooms_llm_last_generation(llm, (dooms_llm_generation_info *) (void *) &old_info) == DOOMS_OK);
                CHECK(old_info.completion_tokens == 2 && old_info.sentinel == 77);
                CHECK(old_info.struct_size == offsetof(dooms_llm_generation_info, finish_reason));
            }
            dooms_llm_close(llm);
        }
    } else {
        printf("c api: LLM checks skipped (set DOOMS_LLM_FILE)\n");
    }

    /* vla */
    dooms_vla * vla = (dooms_vla *) 1;
    dooms_vla_paths ap;
    memset(&ap, 0, sizeof ap);
    ap.struct_size = sizeof ap;
    CHECK(dooms_vla_open(&ap, NULL, &vla) == DOOMS_INVALID_ARGUMENT);
    CHECK(vla == NULL);
    ap.vision = ap.prefix = ap.expert = ap.host = ap.vocab = "/nonexistent/x";
    {
        dooms_code c = dooms_vla_open(&ap, NULL, &vla);
        CHECK(c == DOOMS_NOT_FOUND || c == DOOMS_UNSUPPORTED);
    }
    dooms_vla_info info;
    CHECK(dooms_vla_info_get(NULL, &info) == DOOMS_INVALID_ARGUMENT);
    float actions[4];
    CHECK(dooms_vla_infer(NULL, &img, 1, "x", NULL, 0, NULL, actions, 4) == DOOMS_INVALID_ARGUMENT);

    /* vla runner: defaults, argument validation (the logic itself is tested in C++ with a fake model) */
    {
        dooms_vla_runner_options ro;
        dooms_vla_runner * runner = (dooms_vla_runner *) 1;
        dooms_vla_action_info ai;
        dooms_vla_runner_stats rs;
        float act[4];
        int32_t flag = 7;
        CHECK(dooms_vla_runner_options_init(NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_vla_runner_options_init(&ro) == DOOMS_OK);
        CHECK(ro.struct_size == sizeof ro && ro.control_hz == 30.0 && ro.chunk_size_threshold == 0.5 && ro.new_weight == 0.7);
        {
            double w = 0;
            CHECK(dooms_vla_aggregate_weight("conservative", &w) == DOOMS_OK && w == 0.3);
            CHECK(dooms_vla_aggregate_weight("median", &w) == DOOMS_INVALID_ARGUMENT && strstr(dooms_last_error(), "weighted_average") != NULL);
            CHECK(dooms_vla_aggregate_weight(NULL, &w) == DOOMS_INVALID_ARGUMENT);
        }
        CHECK(dooms_vla_runner_open(NULL, &ro, &runner) != DOOMS_OK);
        CHECK(runner == NULL);
        CHECK(dooms_vla_runner_open(NULL, &ro, NULL) == DOOMS_INVALID_ARGUMENT);
        dooms_vla_runner_close(NULL);
        CHECK(dooms_vla_runner_wants_observation(NULL, &flag) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_vla_runner_submit(NULL, &img, 1, "x", NULL, 0, 0.0, &flag) == DOOMS_INVALID_ARGUMENT);
        CHECK(flag == 0);
        memset(&ai, 0, sizeof ai);
        ai.struct_size = sizeof ai;
        CHECK(dooms_vla_runner_next_action(NULL, 0.0, act, 4, &ai) == DOOMS_INVALID_ARGUMENT);
        flag = 7;
        CHECK(dooms_vla_runner_wait_idle(NULL, 1.0, &flag) == DOOMS_INVALID_ARGUMENT);
        CHECK(flag == 0);
        memset(&rs, 0, sizeof rs);
        rs.struct_size = sizeof rs;
        CHECK(dooms_vla_runner_stats_get(NULL, &rs) == DOOMS_INVALID_ARGUMENT);
    }

    /* vla runner on the real model (opt-in) */
    {
        const char * dir = getenv("DOOMS_VLA_DIR");
        if (!dir) {
            printf("        (vla runner on the real model skipped: set DOOMS_VLA_DIR)\n");
        } else {
            char v[1024], pr[1024], ex[1024], ho[1024], vo[1024];
            dooms_vla * model = NULL;
            dooms_vla_runner * runner = NULL;
            dooms_vla_runner * second = (dooms_vla_runner *) 1;
            dooms_vla_runner_options ro;
            dooms_vla_action_info ai;
            dooms_vla_runner_stats rs;
            dooms_vla_info vi;
            static uint8_t pixels[3][64 * 48 * 3];
            dooms_image cams[3];
            float state[6] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f};
            float act[32], chunk[50 * 32];
            int32_t flag = 0, i, k;
            int64_t expect_step = 0;
            snprintf(v, sizeof v, "%s/vision-b1-fp16.mlpackage", dir);
            snprintf(pr, sizeof pr, "%s/prefix-fp16.mlpackage", dir);
            snprintf(ex, sizeof ex, "%s/expert-fp16.mlpackage", dir);
            snprintf(ho, sizeof ho, "%s/host.gguf", dir);
            snprintf(vo, sizeof vo, "%s/vocab.gguf", dir);
            ap.vision = v; ap.prefix = pr; ap.expert = ex; ap.host = ho; ap.vocab = vo;
            CHECK(dooms_vla_open(&ap, NULL, &model) == DOOMS_OK);
            CHECK(dooms_vla_info_get(model, &vi) == DOOMS_OK && vi.n_cameras == 3 && vi.action_dim == 32 && vi.chunk_size == 50);
            for (i = 0; i < 3; i++) {
                for (k = 0; k < 64 * 48 * 3; k++) pixels[i][k] = (uint8_t) ((k * 7 + i * 31) % 251);
                cams[i].width = 64; cams[i].height = 48; cams[i].rgb = pixels[i];
            }
            dooms_vla_runner_options_init(&ro);
            ro.new_weight = 1.5;
            CHECK(dooms_vla_runner_open(model, &ro, &runner) == DOOMS_INVALID_ARGUMENT); /* out of range */
            CHECK(runner == NULL);
            ro.new_weight = 0.7;
            CHECK(dooms_vla_runner_open(model, &ro, &runner) == DOOMS_OK);
            CHECK(dooms_vla_runner_open(model, &ro, &second) == DOOMS_INVALID_ARGUMENT); /* one runner per handle */
            CHECK(second == NULL);
            CHECK(dooms_vla_infer(model, cams, 3, "x", state, 6, NULL, chunk, 50 * 32) == DOOMS_INVALID_ARGUMENT); /* the runner owns the model */
            CHECK(strstr(dooms_last_error(), "runner") != NULL);
            memset(&ai, 0, sizeof ai);
            ai.struct_size = sizeof ai;
            CHECK(dooms_vla_runner_next_action(runner, 0.0, act, 31, &ai) == DOOMS_RESOURCE_EXHAUSTED);
            CHECK(dooms_vla_runner_next_action(runner, 0.0, act, 32, &ai) == DOOMS_OK && ai.available == 0 && ai.step == -1);
            CHECK(dooms_vla_runner_submit(runner, cams, 2, "x", state, 6, 0.0, &flag) == DOOMS_INVALID_ARGUMENT);
            CHECK(dooms_vla_runner_wants_observation(runner, &flag) == DOOMS_OK && flag == 1);
            CHECK(dooms_vla_runner_submit(runner, cams, 3, "pick up the cube", state, 6, 0.0, &flag) == DOOMS_OK && flag == 1);
            CHECK(dooms_vla_runner_submit(runner, cams, 3, "pick up the cube", state, 6, 0.0, &flag) == DOOMS_OK && flag == 0); /* in flight */
            CHECK(dooms_vla_runner_wait_idle(runner, -1.0, &flag) == DOOMS_INVALID_ARGUMENT);
            CHECK(dooms_vla_runner_wait_idle(runner, 600.0, &flag) == DOOMS_OK && flag == 1);
            /* the handle may be closed first: the runner keeps the model */
            dooms_vla_close(model);
            /* 80 steps on a synthetic 30 Hz clock; each requested chunk is waited for before the next step (zero latency in steps) */
            for (k = 0; k < 80; k++) {
                const double now = k / 30.0;
                CHECK(dooms_vla_runner_next_action(runner, now, act, 32, &ai) == DOOMS_OK);
                CHECK(ai.available == 1 && ai.step == expect_step && ai.skipped == 0);
                expect_step++;
                CHECK(dooms_vla_runner_wants_observation(runner, &flag) == DOOMS_OK);
                if (flag) {
                    CHECK(dooms_vla_runner_submit(runner, cams, 3, "pick up the cube", state, 6, now, &flag) == DOOMS_OK && flag == 1);
                    CHECK(dooms_vla_runner_wait_idle(runner, 600.0, &flag) == DOOMS_OK && flag == 1);
                }
            }
            memset(&rs, 0, sizeof rs);
            rs.struct_size = sizeof rs;
            CHECK(dooms_vla_runner_stats_get(runner, &rs) == DOOMS_OK);
            CHECK(rs.actions_returned == 80 && rs.stalled_calls == 0 && rs.inferences_failed == 0 && rs.last_error_code == DOOMS_OK && strlen(rs.last_error) == 0);
            CHECK(rs.chunks_received == 4 && rs.blended == 3 * 25 && rs.stale_dropped == 3); /* observations at steps 24, 48, 72 */
            printf("        vla runner: %lld actions, %lld chunks, %lld blended, last inference %.1f ms\n", (long long) rs.actions_returned, (long long) rs.chunks_received,
                   (long long) rs.blended, rs.last_inference_ms);
            CHECK(dooms_vla_runner_next_action(runner, 0.0, act, 32, &ai) == DOOMS_INVALID_ARGUMENT); /* time went backwards */
            dooms_vla_runner_close(runner);
        }
    }

    /* speech to text: argument checks always; with DOOMS_WHISPER_DIR (converted model) and DOOMS_WHISPER_CLIP (a speech file) the real model (CTest c_api_whisper) */
    {
        dooms_whisper * w = (dooms_whisper *) 1;
        dooms_whisper_paths wp;
        dooms_whisper_options wo;
        dooms_whisper_timings wt;
        char small[4], text[4096];
        size_t needed = 0;
        float silence[16000];
        CHECK(dooms_whisper_open(NULL, NULL, &w) == DOOMS_INVALID_ARGUMENT && w == NULL);
        memset(&wp, 0, sizeof wp);
        CHECK(dooms_whisper_open(&wp, NULL, &w) == DOOMS_INVALID_ARGUMENT); /* struct_size unset */
        wp.struct_size = sizeof wp;
        CHECK(dooms_whisper_open(&wp, NULL, &w) == DOOMS_INVALID_ARGUMENT); /* no paths */
        wp.encoder = "/nonexistent/encoder-fp16.mlpackage";
        wp.decoder = "/nonexistent/decoder.gguf";
        CHECK(dooms_whisper_open(&wp, NULL, &w) == DOOMS_NOT_FOUND && w == NULL && strlen(dooms_last_error()) > 0);
        CHECK(dooms_whisper_open(&wp, NULL, NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_whisper_transcribe(NULL, silence, 16000, text, sizeof text, NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_whisper_transcribe_file(NULL, "a.wav", text, sizeof text, NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_whisper_last_text(NULL, text, sizeof text, NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_whisper_sample_rate(NULL) == 0);
        dooms_whisper_close(NULL);
        if (getenv("DOOMS_WHISPER_DIR") && *getenv("DOOMS_WHISPER_DIR") && getenv("DOOMS_WHISPER_CLIP")) {
            char enc[1024], dec[1024];
            snprintf(enc, sizeof enc, "%s/encoder-fp16.mlpackage", getenv("DOOMS_WHISPER_DIR"));
            snprintf(dec, sizeof dec, "%s/decoder.gguf", getenv("DOOMS_WHISPER_DIR"));
            wp.encoder = enc;
            wp.decoder = dec;
            memset(&wo, 0, sizeof wo);
            wo.struct_size = sizeof wo;
            wo.language = "xx";
            CHECK(dooms_whisper_open(&wp, &wo, &w) == DOOMS_INVALID_ARGUMENT); /* unknown language */
            wo.language = NULL;
            wo.max_tokens = -1;
            CHECK(dooms_whisper_open(&wp, &wo, &w) == DOOMS_INVALID_ARGUMENT);
            wo.max_tokens = 0;
            CHECK(dooms_whisper_open(&wp, &wo, &w) == DOOMS_OK);
            if (w) {
                CHECK(dooms_whisper_sample_rate(w) == 16000);
                CHECK(dooms_whisper_last_text(w, text, sizeof text, NULL) == DOOMS_NOT_FOUND);
                CHECK(dooms_whisper_transcribe(w, NULL, 0, text, sizeof text, NULL) == DOOMS_INVALID_ARGUMENT);
                CHECK(dooms_whisper_transcribe_file(w, "/nonexistent.wav", text, sizeof text, NULL) == DOOMS_NOT_FOUND);
                /* too small a buffer: the needed size is reported and the text is kept for dooms_whisper_last_text */
                CHECK(dooms_whisper_transcribe_file(w, getenv("DOOMS_WHISPER_CLIP"), small, sizeof small, &needed) == DOOMS_RESOURCE_EXHAUSTED);
                CHECK(needed > sizeof small && needed < sizeof text);
                CHECK(dooms_whisper_last_text(w, text, sizeof text, NULL) == DOOMS_OK && strlen(text) + 1 == needed);
                memset(&wt, 0, sizeof wt);
                wt.struct_size = sizeof wt;
                CHECK(dooms_whisper_last_timings(w, &wt) == DOOMS_OK);
                CHECK(wt.windows == 1 && wt.tokens > 0 && wt.audio_s > 1.0 && wt.total_ms > 0 && wt.encoder_ms > 0 && wt.decode_ms > 0);
                printf("        whisper: \"%s\" (%d tokens, %.0f ms for %.1f s)\n", text, wt.tokens, wt.total_ms, wt.audio_s);
                memset(silence, 0, sizeof silence);
                CHECK(dooms_whisper_transcribe(w, silence, 16000, text, sizeof text, &needed) == DOOMS_OK && strlen(text) + 1 == needed);
                CHECK(dooms_whisper_last_timings(w, &wt) == DOOMS_OK && wt.audio_s == 1.0);
                dooms_whisper_close(w);
            }
        }
    }

    /* text to speech: argument checks always; with DOOMS_TTS_DIR (converted Kokoro) the real model (CTest c_api_tts) */
    {
        dooms_tts * t = (dooms_tts *) 1;
        dooms_tts_paths tp;
        dooms_tts_timings tt;
        float small[8];
        size_t needed = 0;
        CHECK(dooms_tts_open(NULL, NULL, &t) == DOOMS_INVALID_ARGUMENT && t == NULL);
        memset(&tp, 0, sizeof tp);
        CHECK(dooms_tts_open(&tp, NULL, &t) == DOOMS_INVALID_ARGUMENT);
        tp.struct_size = sizeof tp;
        CHECK(dooms_tts_open(&tp, NULL, &t) == DOOMS_INVALID_ARGUMENT);
        tp.prosody = "/nonexistent/prosody-fp32.mlpackage";
        tp.f0n = "/nonexistent/f0n-fp32.mlpackage";
        tp.model = "/nonexistent/kokoro.gguf";
        CHECK(dooms_tts_open(&tp, NULL, &t) == DOOMS_NOT_FOUND && t == NULL);
        CHECK(dooms_tts_synthesize(NULL, "a", "af_heart", 1.f, 0, small, 8, NULL) == DOOMS_INVALID_ARGUMENT);
        CHECK(dooms_tts_sample_rate(NULL) == 0 && dooms_tts_voice_count(NULL) == 0 && dooms_tts_voice_name(NULL, 0) == NULL);
        dooms_tts_close(NULL);
        if (getenv("DOOMS_TTS_DIR") && *getenv("DOOMS_TTS_DIR")) {
            char p1[1024], p2[1024], p3[1024];
            float * audio;
            snprintf(p1, sizeof p1, "%s/prosody-fp32.mlpackage", getenv("DOOMS_TTS_DIR"));
            snprintf(p2, sizeof p2, "%s/f0n-fp32.mlpackage", getenv("DOOMS_TTS_DIR"));
            snprintf(p3, sizeof p3, "%s/kokoro.gguf", getenv("DOOMS_TTS_DIR"));
            tp.prosody = p1; tp.f0n = p2; tp.model = p3;
            CHECK(dooms_tts_open(&tp, NULL, &t) == DOOMS_OK);
            if (t) {
                CHECK(dooms_tts_sample_rate(t) == 24000 && dooms_tts_max_phonemes(t) > 100 && dooms_tts_voice_count(t) > 0);
                CHECK(dooms_tts_voice_name(t, 0) != NULL && dooms_tts_voice_name(t, dooms_tts_voice_count(t)) == NULL);
                CHECK(dooms_tts_last_audio(t, small, 8, NULL) == DOOMS_NOT_FOUND);
                CHECK(dooms_tts_synthesize(t, "h\xC9\x99l\xCB\x88O", "nobody", 1.f, 0, small, 8, NULL) == DOOMS_INVALID_ARGUMENT);
                CHECK(dooms_tts_synthesize(t, "h\xC9\x99l\xCB\x88O", dooms_tts_voice_name(t, 0), 0.f, 0, small, 8, NULL) == DOOMS_INVALID_ARGUMENT);
                /* too small: the needed size is reported, the samples are kept */
                CHECK(dooms_tts_synthesize(t, "h\xC9\x99l\xCB\x88O w\xCB\x88\xC9\x9C\xC9\xB9ld.", dooms_tts_voice_name(t, 0), 1.f, 7, small, 8, &needed) == DOOMS_RESOURCE_EXHAUSTED);
                CHECK(needed > 24000 / 2 && needed % 600 == 0);
                audio = (float *) malloc(needed * sizeof(float));
                CHECK(audio && dooms_tts_last_audio(t, audio, needed, NULL) == DOOMS_OK);
                memset(&tt, 0, sizeof tt);
                tt.struct_size = sizeof tt;
                CHECK(dooms_tts_last_timings(t, &tt) == DOOMS_OK && tt.frames * 600 == (int32_t) needed && tt.total_ms > 0 && tt.vocoder_ms > 0);
                printf("        tts: %.2f s of audio in %.0f ms\n", tt.audio_s, tt.total_ms);
                free(audio);
                dooms_tts_close(t);
            }
        }
    }

    /* the last error is per call: a successful call clears it */
    CHECK(dooms_c_api_version() == DOOMS_C_API_VERSION);
    printf("%s\n", failures ? "FAILED" : "c api: ok");
    return failures ? 1 : 0;
}
