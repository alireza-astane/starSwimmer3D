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
    sphere { m*<0.3672907277596156,0.9021888648220138,0.08475547270607663>, 1 }        
    sphere {  m*<0.6080258325013075,1.0308989430023394,3.07231024382663>, 1 }
    sphere {  m*<3.1019991217658727,1.0042228402083886,-1.1444540527451066>, 1 }
    sphere {  m*<-1.2543246321332737,3.2306628092406164,-0.8891902927098928>, 1}
    sphere { m*<-3.53579579655952,-6.4760366233696,-2.176668228864459>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6080258325013075,1.0308989430023394,3.07231024382663>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5 }
    cylinder { m*<3.1019991217658727,1.0042228402083886,-1.1444540527451066>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5}
    cylinder { m*<-1.2543246321332737,3.2306628092406164,-0.8891902927098928>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5 }
    cylinder {  m*<-3.53579579655952,-6.4760366233696,-2.176668228864459>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5}

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
    sphere { m*<0.3672907277596156,0.9021888648220138,0.08475547270607663>, 1 }        
    sphere {  m*<0.6080258325013075,1.0308989430023394,3.07231024382663>, 1 }
    sphere {  m*<3.1019991217658727,1.0042228402083886,-1.1444540527451066>, 1 }
    sphere {  m*<-1.2543246321332737,3.2306628092406164,-0.8891902927098928>, 1}
    sphere { m*<-3.53579579655952,-6.4760366233696,-2.176668228864459>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6080258325013075,1.0308989430023394,3.07231024382663>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5 }
    cylinder { m*<3.1019991217658727,1.0042228402083886,-1.1444540527451066>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5}
    cylinder { m*<-1.2543246321332737,3.2306628092406164,-0.8891902927098928>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5 }
    cylinder {  m*<-3.53579579655952,-6.4760366233696,-2.176668228864459>, <0.3672907277596156,0.9021888648220138,0.08475547270607663>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    