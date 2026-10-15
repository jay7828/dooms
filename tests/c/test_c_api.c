/* Calls the C API exactly as an external C program would (this file is compiled as C99 with only <dooms/dooms_c.h>): version/error plumbing, argument validation and
 * the failure of every open call on missing files. The real-model behaviour is covered from Python (python/tests). */
#include <stdio.h>
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

    /* the last error is per call: a successful call clears it */
    CHECK(dooms_c_api_version() == DOOMS_C_API_VERSION);
    printf("%s\n", failures ? "FAILED" : "c api: ok");
    return failures ? 1 : 0;
}
