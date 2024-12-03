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
    sphere { m*<1.0226660311895064,0.435707527890546,0.470535480200589>, 1 }        
    sphere {  m*<1.2666096646561849,0.47068426823423515,3.4603948945619516>, 1 }
    sphere {  m*<3.7598568537187202,0.470684268234235,-0.7568873139286665>, 1 }
    sphere {  m*<-2.903973552050006,6.6179204547329675,-1.851146438884274>, 1}
    sphere { m*<-3.803746075952643,-7.844178228009605,-2.3824770115695726>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2666096646561849,0.47068426823423515,3.4603948945619516>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5 }
    cylinder { m*<3.7598568537187202,0.470684268234235,-0.7568873139286665>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5}
    cylinder { m*<-2.903973552050006,6.6179204547329675,-1.851146438884274>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5 }
    cylinder {  m*<-3.803746075952643,-7.844178228009605,-2.3824770115695726>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5}

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
    sphere { m*<1.0226660311895064,0.435707527890546,0.470535480200589>, 1 }        
    sphere {  m*<1.2666096646561849,0.47068426823423515,3.4603948945619516>, 1 }
    sphere {  m*<3.7598568537187202,0.470684268234235,-0.7568873139286665>, 1 }
    sphere {  m*<-2.903973552050006,6.6179204547329675,-1.851146438884274>, 1}
    sphere { m*<-3.803746075952643,-7.844178228009605,-2.3824770115695726>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.2666096646561849,0.47068426823423515,3.4603948945619516>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5 }
    cylinder { m*<3.7598568537187202,0.470684268234235,-0.7568873139286665>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5}
    cylinder { m*<-2.903973552050006,6.6179204547329675,-1.851146438884274>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5 }
    cylinder {  m*<-3.803746075952643,-7.844178228009605,-2.3824770115695726>, <1.0226660311895064,0.435707527890546,0.470535480200589>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    