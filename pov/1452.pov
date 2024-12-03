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
    sphere { m*<0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 1 }        
    sphere {  m*<0.713576629695301,-2.248225087029796e-18,3.9565510656736347>, 1 }
    sphere {  m*<6.983492358776157,1.921342636718585e-18,-1.5190890491728248>, 1 }
    sphere {  m*<-4.196586732401547,8.164965809277259,-2.225986055915233>, 1}
    sphere { m*<-4.196586732401547,-8.164965809277259,-2.2259860559152367>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.713576629695301,-2.248225087029796e-18,3.9565510656736347>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5 }
    cylinder { m*<6.983492358776157,1.921342636718585e-18,-1.5190890491728248>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5}
    cylinder { m*<-4.196586732401547,8.164965809277259,-2.225986055915233>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5 }
    cylinder {  m*<-4.196586732401547,-8.164965809277259,-2.2259860559152367>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5}

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
    sphere { m*<0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 1 }        
    sphere {  m*<0.713576629695301,-2.248225087029796e-18,3.9565510656736347>, 1 }
    sphere {  m*<6.983492358776157,1.921342636718585e-18,-1.5190890491728248>, 1 }
    sphere {  m*<-4.196586732401547,8.164965809277259,-2.225986055915233>, 1}
    sphere { m*<-4.196586732401547,-8.164965809277259,-2.2259860559152367>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.713576629695301,-2.248225087029796e-18,3.9565510656736347>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5 }
    cylinder { m*<6.983492358776157,1.921342636718585e-18,-1.5190890491728248>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5}
    cylinder { m*<-4.196586732401547,8.164965809277259,-2.225986055915233>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5 }
    cylinder {  m*<-4.196586732401547,-8.164965809277259,-2.2259860559152367>, <0.6195847782368111,-5.596811203802524e-18,0.9580205601393551>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    