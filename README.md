# ChatGPT-Platform
Clone version of the ChatGPT platform for Generating the Text , image from text and text from image.

## The API are created on the FastAPI framework and DataBase is used for storing the conversation and file is SQLite3

"/signup" There is the Sign up functionality which allow user to register and then an email verification mail sent on the user registered email address. "/verify-otp" User have to verify the email address by putting the OTP within 3 mins (lib totp: created a secrect key for verify the otp)

"/login" Then User have to login to get the access token which allow them to auth to there resources.

## There are several APIs endpointa which allow user to create a text conversation , image generation and manage there history.
"/vision"  
"/stream/chat"  
"/stream/chat/{chat_id}" while the conversation is created 


Apart from this I also provided the functionality of the User forgotten password   
"/forgot-passwd"  
"/update-passwd"

change password  
/change-password/{user_id}"

and Take the Subscription of the Service    
"/update-plan/{user_id}" and 

user can also delete the account   
"/delete-user/{user_id}".

There is prebuilt get API which shows the streaming of the text Content.   
"/stream"
