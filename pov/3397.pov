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
    sphere { m*<0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 1 }        
    sphere {  m*<0.4601913678882571,0.7514390826982069,2.986655887946505>, 1 }
    sphere {  m*<2.954164657152822,0.7247629799042559,-1.23010840862523>, 1 }
    sphere {  m*<-1.4021590967463253,2.9512029489364826,-0.9748446485900161>, 1}
    sphere { m*<-3.0360491347535628,-5.531337207970754,-1.8871181666900396>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4601913678882571,0.7514390826982069,2.986655887946505>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5 }
    cylinder { m*<2.954164657152822,0.7247629799042559,-1.23010840862523>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5}
    cylinder { m*<-1.4021590967463253,2.9512029489364826,-0.9748446485900161>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5 }
    cylinder {  m*<-3.0360491347535628,-5.531337207970754,-1.8871181666900396>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5}

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
    sphere { m*<0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 1 }        
    sphere {  m*<0.4601913678882571,0.7514390826982069,2.986655887946505>, 1 }
    sphere {  m*<2.954164657152822,0.7247629799042559,-1.23010840862523>, 1 }
    sphere {  m*<-1.4021590967463253,2.9512029489364826,-0.9748446485900161>, 1}
    sphere { m*<-3.0360491347535628,-5.531337207970754,-1.8871181666900396>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4601913678882571,0.7514390826982069,2.986655887946505>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5 }
    cylinder { m*<2.954164657152822,0.7247629799042559,-1.23010840862523>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5}
    cylinder { m*<-1.4021590967463253,2.9512029489364826,-0.9748446485900161>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5 }
    cylinder {  m*<-3.0360491347535628,-5.531337207970754,-1.8871181666900396>, <0.21945626314656536,0.6227290045178815,-0.0008988831740461001>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    