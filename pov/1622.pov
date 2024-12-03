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
    sphere { m*<0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 1 }        
    sphere {  m*<0.9761234243989715,2.537091236785997e-18,3.8557042270254787>, 1 }
    sphere {  m*<6.064197973457043,5.918505761518533e-18,-1.2618603524071805>, 1 }
    sphere {  m*<-4.0221940747791045,8.164965809277259,-2.25579044653825>, 1}
    sphere { m*<-4.0221940747791045,-8.164965809277259,-2.255790446538253>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9761234243989715,2.537091236785997e-18,3.8557042270254787>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5 }
    cylinder { m*<6.064197973457043,5.918505761518533e-18,-1.2618603524071805>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5}
    cylinder { m*<-4.0221940747791045,8.164965809277259,-2.25579044653825>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5 }
    cylinder {  m*<-4.0221940747791045,-8.164965809277259,-2.255790446538253>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5}

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
    sphere { m*<0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 1 }        
    sphere {  m*<0.9761234243989715,2.537091236785997e-18,3.8557042270254787>, 1 }
    sphere {  m*<6.064197973457043,5.918505761518533e-18,-1.2618603524071805>, 1 }
    sphere {  m*<-4.0221940747791045,8.164965809277259,-2.25579044653825>, 1}
    sphere { m*<-4.0221940747791045,-8.164965809277259,-2.255790446538253>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9761234243989715,2.537091236785997e-18,3.8557042270254787>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5 }
    cylinder { m*<6.064197973457043,5.918505761518533e-18,-1.2618603524071805>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5}
    cylinder { m*<-4.0221940747791045,8.164965809277259,-2.25579044653825>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5 }
    cylinder {  m*<-4.0221940747791045,-8.164965809277259,-2.255790446538253>, <0.8391430081347584,-2.004851391824598e-18,0.8588280329898392>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    