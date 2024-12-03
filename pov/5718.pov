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
    sphere { m*<-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 1 }        
    sphere {  m*<0.09986172098425072,0.28160944148408007,8.660885864864166>, 1 }
    sphere {  m*<5.951967543427925,0.07736341214811554,-4.896350413680912>, 1 }
    sphere {  m*<-2.862201959647072,2.1566174125226967,-2.1424701551911047>, 1}
    sphere { m*<-2.5944147386092404,-2.7310745298812007,-1.9529238700285343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09986172098425072,0.28160944148408007,8.660885864864166>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5 }
    cylinder { m*<5.951967543427925,0.07736341214811554,-4.896350413680912>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5}
    cylinder { m*<-2.862201959647072,2.1566174125226967,-2.1424701551911047>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5 }
    cylinder {  m*<-2.5944147386092404,-2.7310745298812007,-1.9529238700285343>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5}

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
    sphere { m*<-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 1 }        
    sphere {  m*<0.09986172098425072,0.28160944148408007,8.660885864864166>, 1 }
    sphere {  m*<5.951967543427925,0.07736341214811554,-4.896350413680912>, 1 }
    sphere {  m*<-2.862201959647072,2.1566174125226967,-2.1424701551911047>, 1}
    sphere { m*<-2.5944147386092404,-2.7310745298812007,-1.9529238700285343>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.09986172098425072,0.28160944148408007,8.660885864864166>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5 }
    cylinder { m*<5.951967543427925,0.07736341214811554,-4.896350413680912>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5}
    cylinder { m*<-2.862201959647072,2.1566174125226967,-2.1424701551911047>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5 }
    cylinder {  m*<-2.5944147386092404,-2.7310745298812007,-1.9529238700285343>, <-1.198421007291718,-0.17247100625468084,-1.2440437631361374>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    