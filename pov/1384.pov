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
    sphere { m*<0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 1 }        
    sphere {  m*<0.6073834365803038,-6.461901274997655e-19,3.995057264307743>, 1 }
    sphere {  m*<7.350904694607813,3.0968620188970214e-18,-1.6173402051147436>, 1 }
    sphere {  m*<-4.269753412795626,8.164965809277259,-2.2135641289161816>, 1}
    sphere { m*<-4.269753412795626,-8.164965809277259,-2.2135641289161843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6073834365803038,-6.461901274997655e-19,3.995057264307743>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5 }
    cylinder { m*<7.350904694607813,3.0968620188970214e-18,-1.6173402051147436>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5}
    cylinder { m*<-4.269753412795626,8.164965809277259,-2.2135641289161816>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5 }
    cylinder {  m*<-4.269753412795626,-8.164965809277259,-2.2135641289161843>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5}

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
    sphere { m*<0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 1 }        
    sphere {  m*<0.6073834365803038,-6.461901274997655e-19,3.995057264307743>, 1 }
    sphere {  m*<7.350904694607813,3.0968620188970214e-18,-1.6173402051147436>, 1 }
    sphere {  m*<-4.269753412795626,8.164965809277259,-2.2135641289161816>, 1}
    sphere { m*<-4.269753412795626,-8.164965809277259,-2.2135641289161843>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6073834365803038,-6.461901274997655e-19,3.995057264307743>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5 }
    cylinder { m*<7.350904694607813,3.0968620188970214e-18,-1.6173402051147436>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5}
    cylinder { m*<-4.269753412795626,8.164965809277259,-2.2135641289161816>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5 }
    cylinder {  m*<-4.269753412795626,-8.164965809277259,-2.2135641289161843>, <0.5293764682577502,-4.771386186849908e-18,0.9960689661693786>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    