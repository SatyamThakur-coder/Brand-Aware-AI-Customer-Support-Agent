# Pipeline Failure Analysis Report

This report presents empirical analysis of **39 identified failure cases** out of 50 evaluated test conversations.

## Top 5 Failure Categories Analyzed
1. **Ambiguous Intents / Classification Error**
2. **Poor Retrieval Quality**
3. **Incorrect Escalation (Over- or Under-escalation)**
4. **Hallucinated Policy / Low Groundedness**
5. **Insufficient Historical Evidence**

---

### Failure Case #1: Incorrect Escalation
- **Conversation ID**: `conv_205981`
- **Customer Message**: *"[BRAND] [BRAND] and what about this one 👉order id #4__credit_card__"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Generated Draft Reply**: *"Hello! [USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #2: Incorrect Escalation
- **Conversation ID**: `conv_205981`
- **Customer Message**: *"[BRAND] [BRAND] and what about this one 👉order id #4__credit_card__"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Generated Draft Reply**: *"Hello! [USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #3: Incorrect Escalation
- **Conversation ID**: `conv_205981`
- **Customer Message**: *"[BRAND] [BRAND] and what about this one 👉order id #4__credit_card__"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Generated Draft Reply**: *"Hello! [USER] I couldn't comprehend your concern. Could you please elaborate and we will assist you accordingly. Please don't provide your order details, we consider it to be personal information. Our page is visible to the public. ^VH"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #4: Incorrect Escalation
- **Conversation ID**: `conv_179426`
- **Customer Message**: *"[BRAND] I'll try this and make an inform as soon"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] Do not worry, Jay. If you've shared your details in the link, we'll check and get back to you with an update soon. ^MK"*
- **Generated Draft Reply**: *"Hello! [USER] Do not worry, Jay. If you've shared your details in the link, we'll check and get back to you with an update soon. ^MK"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #5: Incorrect Escalation
- **Conversation ID**: `conv_69631`
- **Customer Message**: *"[BRAND] Order# 407-8809404-0041923 why prime order taking 7 days after dispatch? It was mentioned next day delivery, now it's showing next Wednesday. Immediately rectify it. I want it today."*
- **Predicted Intent**: `delivery_issue` | **Expected Intent**: `delivery_issue`
- **Retrieved Evidence Sample**: *"[USER] Hi, during this busy period of Black Friday, some deliveries may take slightly longer than usual. Apologies if this has caused any inconvenience. Please check out this link for more info: [URL] ^JC"*
- **Generated Draft Reply**: *"Hello! [USER] Hi, during this busy period of Black Friday, some deliveries may take slightly longer than usual. Apologies if this has caused any inconvenience. Please check out this link for more info: [URL] ^JC"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #6: Incorrect Escalation
- **Conversation ID**: `conv_232401`
- **Customer Message**: *"En fait #Amazonprime c'est surtout de la merde non? Ça existe un livreur qui fait son taf à Paris ou c'est une utopie? [BRAND] [USER]"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] Dans un souci permanent de qualité et d’efficacité, je vous invite à nous communiquer le numéro de suivi de votre commande. ^SB"*
- **Generated Draft Reply**: *"Hello! [USER] Dans un souci permanent de qualité et d’efficacité, je vous invite à nous communiquer le numéro de suivi de votre commande. ^SB"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #7: Incorrect Escalation
- **Conversation ID**: `conv_76053`
- **Customer Message**: *"[BRAND] It was originally meant to be 28th and then changed to 29th and last night was changed again."*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] Sorry about the delay. Could you kindly let us know the revised date? ^NS"*
- **Generated Draft Reply**: *"Hello! [USER] Sorry about the delay. Could you kindly let us know the revised date? ^NS"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #8: Incorrect Escalation
- **Conversation ID**: `conv_77589`
- **Customer Message**: *"[BRAND] No, doveva essere spedito il 28 novembre. Come posso fare se non riesco a essere presente alla consegna?"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] Nos envie uma foto assim você receber seu pedido, para dividir a felicidade! 🎉 ^CR"*
- **Generated Draft Reply**: *"Hello! [USER] Nos envie uma foto assim você receber seu pedido, para dividir a felicidade! 🎉 ^CR"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #9: Incorrect Escalation
- **Conversation ID**: `conv_191492`
- **Customer Message**: *"[BRAND] Sure..."*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] I’d like to assist you with this. Please fill this form: [URL] and I’ll contact you at the earliest. ^ST"*
- **Generated Draft Reply**: *"Hello! [USER] I’d like to assist you with this. Please fill this form: [URL] and I’ll contact you at the earliest. ^ST"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---

### Failure Case #10: Incorrect Escalation
- **Conversation ID**: `conv_169575`
- **Customer Message**: *"[BRAND] wann wird pretty Little liars Staffel 7 auf deutsch erscheinen"*
- **Predicted Intent**: `other_general` | **Expected Intent**: `other_general`
- **Retrieved Evidence Sample**: *"[USER] Hi, hast du dich über den Link gemeldet? Wie bist du mit uns verblieben? Gruß ^MI"*
- **Generated Draft Reply**: *"Hello! [USER] Hi, hast du dich über den Link gemeldet? Wie bist du mit uns verblieben? Gruß ^MI"*
- **Actual Action**: `ESCALATE` | **Expected Action**: `AUTO_HANDLE`
- **Explanation**: Actual action 'ESCALATE' disagreed with expected gold action 'AUTO_HANDLE'.
- **Hypothesis for Root Cause**: Deterministic safety gate threshold miscalibration or conservative risk boundaries.

---
