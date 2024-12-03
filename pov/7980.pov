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
    sphere { m*<-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 1 }        
    sphere {  m*<1.093880289880894,0.7609866654260935,9.449089636867043>, 1 }
    sphere {  m*<8.46166748820368,0.4758944146338304,-5.121587792206881>, 1 }
    sphere {  m*<-6.434295705485303,6.998975788254465,-3.6307808890252753>, 1}
    sphere { m*<-4.367909277858463,-9.032993771067826,-2.2722884092366393>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.093880289880894,0.7609866654260935,9.449089636867043>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5 }
    cylinder { m*<8.46166748820368,0.4758944146338304,-5.121587792206881>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5}
    cylinder { m*<-6.434295705485303,6.998975788254465,-3.6307808890252753>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5 }
    cylinder {  m*<-4.367909277858463,-9.032993771067826,-2.2722884092366393>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5}

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
    sphere { m*<-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 1 }        
    sphere {  m*<1.093880289880894,0.7609866654260935,9.449089636867043>, 1 }
    sphere {  m*<8.46166748820368,0.4758944146338304,-5.121587792206881>, 1 }
    sphere {  m*<-6.434295705485303,6.998975788254465,-3.6307808890252753>, 1}
    sphere { m*<-4.367909277858463,-9.032993771067826,-2.2722884092366393>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.093880289880894,0.7609866654260935,9.449089636867043>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5 }
    cylinder { m*<8.46166748820368,0.4758944146338304,-5.121587792206881>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5}
    cylinder { m*<-6.434295705485303,6.998975788254465,-3.6307808890252753>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5 }
    cylinder {  m*<-4.367909277858463,-9.032993771067826,-2.2722884092366393>, <-0.32528720431926694,-0.22895224845382356,-0.4002004601680987>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    