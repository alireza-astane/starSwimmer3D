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
    sphere { m*<0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 1 }        
    sphere {  m*<1.0112024316807404,7.955474688322382e-19,3.8415520324417205>, 1 }
    sphere {  m*<5.9397225137848615,4.191742705619943e-18,-1.2255845763512137>, 1 }
    sphere {  m*<-3.9996238027912825,8.164965809277259,-2.259691839694683>, 1}
    sphere { m*<-3.9996238027912825,-8.164965809277259,-2.2596918396946855>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0112024316807404,7.955474688322382e-19,3.8415520324417205>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5 }
    cylinder { m*<5.9397225137848615,4.191742705619943e-18,-1.2255845763512137>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5}
    cylinder { m*<-3.9996238027912825,8.164965809277259,-2.259691839694683>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5 }
    cylinder {  m*<-3.9996238027912825,-8.164965809277259,-2.2596918396946855>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5}

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
    sphere { m*<0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 1 }        
    sphere {  m*<1.0112024316807404,7.955474688322382e-19,3.8415520324417205>, 1 }
    sphere {  m*<5.9397225137848615,4.191742705619943e-18,-1.2255845763512137>, 1 }
    sphere {  m*<-3.9996238027912825,8.164965809277259,-2.259691839694683>, 1}
    sphere { m*<-3.9996238027912825,-8.164965809277259,-2.2596918396946855>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0112024316807404,7.955474688322382e-19,3.8415520324417205>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5 }
    cylinder { m*<5.9397225137848615,4.191742705619943e-18,-1.2255845763512137>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5}
    cylinder { m*<-3.9996238027912825,8.164965809277259,-2.259691839694683>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5 }
    cylinder {  m*<-3.9996238027912825,-8.164965809277259,-2.2596918396946855>, <0.8680834989252636,-3.4790165473809675e-18,0.8449624426217637>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    