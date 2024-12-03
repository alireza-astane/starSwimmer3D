#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 1 }        
    sphere {  m*<1.0072158487692047,0.5722484345942733,9.40895641321502>, 1 }
    sphere {  m*<8.375003047092003,0.2871561838020118,-5.161721015858905>, 1 }
    sphere {  m*<-6.520960146596992,6.81023755742265,-3.6709141126772984>, 1}
    sphere { m*<-3.981357088288304,-8.191158579520078,-2.093280903064587>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0072158487692047,0.5722484345942733,9.40895641321502>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5 }
    cylinder { m*<8.375003047092003,0.2871561838020118,-5.161721015858905>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5}
    cylinder { m*<-6.520960146596992,6.81023755742265,-3.6709141126772984>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5 }
    cylinder {  m*<-3.981357088288304,-8.191158579520078,-2.093280903064587>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 1 }        
    sphere {  m*<1.0072158487692047,0.5722484345942733,9.40895641321502>, 1 }
    sphere {  m*<8.375003047092003,0.2871561838020118,-5.161721015858905>, 1 }
    sphere {  m*<-6.520960146596992,6.81023755742265,-3.6709141126772984>, 1}
    sphere { m*<-3.981357088288304,-8.191158579520078,-2.093280903064587>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0072158487692047,0.5722484345942733,9.40895641321502>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5 }
    cylinder { m*<8.375003047092003,0.2871561838020118,-5.161721015858905>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5}
    cylinder { m*<-6.520960146596992,6.81023755742265,-3.6709141126772984>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5 }
    cylinder {  m*<-3.981357088288304,-8.191158579520078,-2.093280903064587>, <-0.4119516454309562,-0.4176904792856435,-0.4403336838201221>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    