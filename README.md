# ChatGPT-Platform
Clone version of the ChatGPT platform for Generating the Text , image from text and text from image.

## The API are created on the FastAPI framework and DataBase is used for storing the conversation and file is SQLite3

### There is the Sign up functionality which allow user to register and then an email verification mail sent on the user registered email address. User have to verify the email address by putting the OTP within 3 mins (lib totp: created a secrect key for verify the otp)

# Then User have to login to get the access token which allow them to auth to there resources.

## There are several APIs endpointa which allow user to create a text conversation , image generation and manage there history.
# Apart from this I also provided the functionality of the User forgotten password , change password and Take the Subscription of the Service.