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
    sphere { m*<1.2100929619180825,0.12124746496953588,0.5813551594170938>, 1 }        
    sphere {  m*<1.4543206286759252,0.13021674122742063,3.5713835955905573>, 1 }
    sphere {  m*<3.9475678177384625,0.13021674122742063,-0.6458986129000603>, 1 }
    sphere {  m*<-3.480666649915324,7.743675936045955,-2.192133725037518>, 1}
    sphere { m*<-3.722794865921666,-8.073498413108,-2.3346091322363645>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4543206286759252,0.13021674122742063,3.5713835955905573>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5 }
    cylinder { m*<3.9475678177384625,0.13021674122742063,-0.6458986129000603>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5}
    cylinder { m*<-3.480666649915324,7.743675936045955,-2.192133725037518>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5 }
    cylinder {  m*<-3.722794865921666,-8.073498413108,-2.3346091322363645>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5}

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
    sphere { m*<1.2100929619180825,0.12124746496953588,0.5813551594170938>, 1 }        
    sphere {  m*<1.4543206286759252,0.13021674122742063,3.5713835955905573>, 1 }
    sphere {  m*<3.9475678177384625,0.13021674122742063,-0.6458986129000603>, 1 }
    sphere {  m*<-3.480666649915324,7.743675936045955,-2.192133725037518>, 1}
    sphere { m*<-3.722794865921666,-8.073498413108,-2.3346091322363645>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.4543206286759252,0.13021674122742063,3.5713835955905573>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5 }
    cylinder { m*<3.9475678177384625,0.13021674122742063,-0.6458986129000603>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5}
    cylinder { m*<-3.480666649915324,7.743675936045955,-2.192133725037518>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5 }
    cylinder {  m*<-3.722794865921666,-8.073498413108,-2.3346091322363645>, <1.2100929619180825,0.12124746496953588,0.5813551594170938>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    