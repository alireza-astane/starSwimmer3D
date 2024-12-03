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
    sphere { m*<0.34817139722947305,0.8660465115957385,0.07367785325039013>, 1 }        
    sphere {  m*<0.5889065019711649,0.9947565897760642,3.0612326243709425>, 1 }
    sphere {  m*<3.08287979123573,0.9680804869821131,-1.1555316722007931>, 1 }
    sphere {  m*<-1.2734439626634169,3.1945204560143416,-0.9002679121655793>, 1}
    sphere { m*<-3.472782886665788,-6.356919751384765,-2.140158946518106>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5889065019711649,0.9947565897760642,3.0612326243709425>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5 }
    cylinder { m*<3.08287979123573,0.9680804869821131,-1.1555316722007931>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5}
    cylinder { m*<-1.2734439626634169,3.1945204560143416,-0.9002679121655793>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5 }
    cylinder {  m*<-3.472782886665788,-6.356919751384765,-2.140158946518106>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5}

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
    sphere { m*<0.34817139722947305,0.8660465115957385,0.07367785325039013>, 1 }        
    sphere {  m*<0.5889065019711649,0.9947565897760642,3.0612326243709425>, 1 }
    sphere {  m*<3.08287979123573,0.9680804869821131,-1.1555316722007931>, 1 }
    sphere {  m*<-1.2734439626634169,3.1945204560143416,-0.9002679121655793>, 1}
    sphere { m*<-3.472782886665788,-6.356919751384765,-2.140158946518106>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5889065019711649,0.9947565897760642,3.0612326243709425>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5 }
    cylinder { m*<3.08287979123573,0.9680804869821131,-1.1555316722007931>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5}
    cylinder { m*<-1.2734439626634169,3.1945204560143416,-0.9002679121655793>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5 }
    cylinder {  m*<-3.472782886665788,-6.356919751384765,-2.140158946518106>, <0.34817139722947305,0.8660465115957385,0.07367785325039013>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    