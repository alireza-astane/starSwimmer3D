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
    sphere { m*<0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 1 }        
    sphere {  m*<1.1380669317908374,9.895811267701506e-19,3.7888775243306485>, 1 }
    sphere {  m*<5.484840910280056,5.6474688986571864e-18,-1.0895983580789699>, 1 }
    sphere {  m*<-3.9194861068251283,8.164965809277259,-2.27369267362426>, 1}
    sphere { m*<-3.9194861068251283,-8.164965809277259,-2.2736926736242626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1380669317908374,9.895811267701506e-19,3.7888775243306485>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5 }
    cylinder { m*<5.484840910280056,5.6474688986571864e-18,-1.0895983580789699>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5}
    cylinder { m*<-3.9194861068251283,8.164965809277259,-2.27369267362426>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5 }
    cylinder {  m*<-3.9194861068251283,-8.164965809277259,-2.2736926736242626>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5}

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
    sphere { m*<0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 1 }        
    sphere {  m*<1.1380669317908374,9.895811267701506e-19,3.7888775243306485>, 1 }
    sphere {  m*<5.484840910280056,5.6474688986571864e-18,-1.0895983580789699>, 1 }
    sphere {  m*<-3.9194861068251283,8.164965809277259,-2.27369267362426>, 1}
    sphere { m*<-3.9194861068251283,-8.164965809277259,-2.2736926736242626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.1380669317908374,9.895811267701506e-19,3.7888775243306485>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5 }
    cylinder { m*<5.484840910280056,5.6474688986571864e-18,-1.0895983580789699>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5}
    cylinder { m*<-3.9194861068251283,8.164965809277259,-2.27369267362426>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5 }
    cylinder {  m*<-3.9194861068251283,-8.164965809277259,-2.2736926736242626>, <0.9719308179004263,-1.6192706588633375e-18,0.7934748276868617>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    