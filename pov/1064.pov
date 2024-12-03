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
    sphere { m*<0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 1 }        
    sphere {  m*<0.10299194538524678,-4.653981303497048e-18,4.163662333366551>, 1 }
    sphere {  m*<9.077771098970826,1.0474645591097425e-18,-2.053293948160836>, 1 }
    sphere {  m*<-4.635868587572477,8.164965809277259,-2.1511257826360604>, 1}
    sphere { m*<-4.635868587572477,-8.164965809277259,-2.151125782636063>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10299194538524678,-4.653981303497048e-18,4.163662333366551>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5 }
    cylinder { m*<9.077771098970826,1.0474645591097425e-18,-2.053293948160836>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5}
    cylinder { m*<-4.635868587572477,8.164965809277259,-2.1511257826360604>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5 }
    cylinder {  m*<-4.635868587572477,-8.164965809277259,-2.151125782636063>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5}

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
    sphere { m*<0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 1 }        
    sphere {  m*<0.10299194538524678,-4.653981303497048e-18,4.163662333366551>, 1 }
    sphere {  m*<9.077771098970826,1.0474645591097425e-18,-2.053293948160836>, 1 }
    sphere {  m*<-4.635868587572477,8.164965809277259,-2.1511257826360604>, 1}
    sphere { m*<-4.635868587572477,-8.164965809277259,-2.151125782636063>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.10299194538524678,-4.653981303497048e-18,4.163662333366551>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5 }
    cylinder { m*<9.077771098970826,1.0474645591097425e-18,-2.053293948160836>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5}
    cylinder { m*<-4.635868587572477,8.164965809277259,-2.1511257826360604>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5 }
    cylinder {  m*<-4.635868587572477,-8.164965809277259,-2.151125782636063>, <0.09122039210348197,-4.717656208380523e-18,1.163685072996225>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    