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
    sphere { m*<-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 1 }        
    sphere {  m*<0.997862216916429,0.551878050788648,9.404624862762603>, 1 }
    sphere {  m*<8.365649415239227,0.26678579999638585,-5.166052566311327>, 1 }
    sphere {  m*<-6.530313778449768,6.789867173617021,-3.6752456631297203>, 1}
    sphere { m*<-3.9390870877030615,-8.099102771793387,-2.0737061920816546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.997862216916429,0.551878050788648,9.404624862762603>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5 }
    cylinder { m*<8.365649415239227,0.26678579999638585,-5.166052566311327>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5}
    cylinder { m*<-6.530313778449768,6.789867173617021,-3.6752456631297203>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5 }
    cylinder {  m*<-3.9390870877030615,-8.099102771793387,-2.0737061920816546>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5}

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
    sphere { m*<-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 1 }        
    sphere {  m*<0.997862216916429,0.551878050788648,9.404624862762603>, 1 }
    sphere {  m*<8.365649415239227,0.26678579999638585,-5.166052566311327>, 1 }
    sphere {  m*<-6.530313778449768,6.789867173617021,-3.6752456631297203>, 1}
    sphere { m*<-3.9390870877030615,-8.099102771793387,-2.0737061920816546>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.997862216916429,0.551878050788648,9.404624862762603>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5 }
    cylinder { m*<8.365649415239227,0.26678579999638585,-5.166052566311327>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5}
    cylinder { m*<-6.530313778449768,6.789867173617021,-3.6752456631297203>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5 }
    cylinder {  m*<-3.9390870877030615,-8.099102771793387,-2.0737061920816546>, <-0.42130527728373257,-0.43806086309126924,-0.44466523427254323>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    