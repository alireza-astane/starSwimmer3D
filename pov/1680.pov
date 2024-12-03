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
    sphere { m*<0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 1 }        
    sphere {  m*<1.0643185995837325,2.0812567516732523e-18,3.8197893080109635>, 1 }
    sphere {  m*<5.75024536084678,5.90320202912765e-18,-1.1696147917144868>, 1 }
    sphere {  m*<-3.9657859248933804,8.164965809277259,-2.2655720320066273>, 1}
    sphere { m*<-3.9657859248933804,-8.164965809277259,-2.26557203200663>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0643185995837325,2.0812567516732523e-18,3.8197893080109635>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5 }
    cylinder { m*<5.75024536084678,5.90320202912765e-18,-1.1696147917144868>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5}
    cylinder { m*<-3.9657859248933804,8.164965809277259,-2.2655720320066273>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5 }
    cylinder {  m*<-3.9657859248933804,-8.164965809277259,-2.26557203200663>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5}

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
    sphere { m*<0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 1 }        
    sphere {  m*<1.0643185995837325,2.0812567516732523e-18,3.8197893080109635>, 1 }
    sphere {  m*<5.75024536084678,5.90320202912765e-18,-1.1696147917144868>, 1 }
    sphere {  m*<-3.9657859248933804,8.164965809277259,-2.2655720320066273>, 1}
    sphere { m*<-3.9657859248933804,-8.164965809277259,-2.26557203200663>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.0643185995837325,2.0812567516732523e-18,3.8197893080109635>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5 }
    cylinder { m*<5.75024536084678,5.90320202912765e-18,-1.1696147917144868>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5}
    cylinder { m*<-3.9657859248933804,8.164965809277259,-2.2655720320066273>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5 }
    cylinder {  m*<-3.9657859248933804,-8.164965809277259,-2.26557203200663>, <0.9117199807314237,-1.3920891577406633e-18,0.823667075170876>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    