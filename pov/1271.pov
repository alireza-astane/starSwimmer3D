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
    sphere { m*<0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 1 }        
    sphere {  m*<0.42995981382088694,-3.131984534225237e-18,4.056857493662038>, 1 }
    sphere {  m*<7.961109761887241,2.370553532378999e-18,-1.775787799211827>, 1 }
    sphere {  m*<-4.395167126333912,8.164965809277259,-2.192266844123017>, 1}
    sphere { m*<-4.395167126333912,-8.164965809277259,-2.1922668441230204>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42995981382088694,-3.131984534225237e-18,4.056857493662038>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5 }
    cylinder { m*<7.961109761887241,2.370553532378999e-18,-1.775787799211827>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5}
    cylinder { m*<-4.395167126333912,8.164965809277259,-2.192266844123017>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5 }
    cylinder {  m*<-4.395167126333912,-8.164965809277259,-2.1922668441230204>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5}

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
    sphere { m*<0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 1 }        
    sphere {  m*<0.42995981382088694,-3.131984534225237e-18,4.056857493662038>, 1 }
    sphere {  m*<7.961109761887241,2.370553532378999e-18,-1.775787799211827>, 1 }
    sphere {  m*<-4.395167126333912,8.164965809277259,-2.192266844123017>, 1}
    sphere { m*<-4.395167126333912,-8.164965809277259,-2.1922668441230204>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.42995981382088694,-3.131984534225237e-18,4.056857493662038>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5 }
    cylinder { m*<7.961109761887241,2.370553532378999e-18,-1.775787799211827>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5}
    cylinder { m*<-4.395167126333912,8.164965809277259,-2.192266844123017>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5 }
    cylinder {  m*<-4.395167126333912,-8.164965809277259,-2.1922668441230204>, <0.376995408231149,-5.78843443421382e-18,1.0573233438122576>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    