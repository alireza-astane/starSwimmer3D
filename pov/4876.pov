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
    sphere { m*<-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 1 }        
    sphere {  m*<0.4824135314634196,0.2627067084783423,7.631115409807039>, 1 }
    sphere {  m*<2.483762551391571,-0.02735311763194708,-2.699181800067702>, 1 }
    sphere {  m*<-1.8725612025075757,2.1990868514002777,-2.443918040032489>, 1}
    sphere { m*<-1.604773981469744,-2.6886050910036197,-2.254371754869916>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4824135314634196,0.2627067084783423,7.631115409807039>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5 }
    cylinder { m*<2.483762551391571,-0.02735311763194708,-2.699181800067702>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5}
    cylinder { m*<-1.8725612025075757,2.1990868514002777,-2.443918040032489>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5 }
    cylinder {  m*<-1.604773981469744,-2.6886050910036197,-2.254371754869916>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5}

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
    sphere { m*<-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 1 }        
    sphere {  m*<0.4824135314634196,0.2627067084783423,7.631115409807039>, 1 }
    sphere {  m*<2.483762551391571,-0.02735311763194708,-2.699181800067702>, 1 }
    sphere {  m*<-1.8725612025075757,2.1990868514002777,-2.443918040032489>, 1}
    sphere { m*<-1.604773981469744,-2.6886050910036197,-2.254371754869916>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4824135314634196,0.2627067084783423,7.631115409807039>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5 }
    cylinder { m*<2.483762551391571,-0.02735311763194708,-2.699181800067702>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5}
    cylinder { m*<-1.8725612025075757,2.1990868514002777,-2.443918040032489>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5 }
    cylinder {  m*<-1.604773981469744,-2.6886050910036197,-2.254371754869916>, <-0.2509458426146858,-0.12938709301832124,-1.4699722746165202>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    