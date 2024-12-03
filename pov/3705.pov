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
    sphere { m*<0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 1 }        
    sphere {  m*<0.25485600565421745,0.36328201935292714,2.867685874847031>, 1 }
    sphere {  m*<2.7488292949187874,0.3366059165589764,-1.3490784217247063>, 1 }
    sphere {  m*<-1.6074944589803657,2.5630458855912037,-1.0938146616894917>, 1}
    sphere { m*<-2.268433341197093,-4.080269603430596,-1.4423664199807393>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25485600565421745,0.36328201935292714,2.867685874847031>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5 }
    cylinder { m*<2.7488292949187874,0.3366059165589764,-1.3490784217247063>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5}
    cylinder { m*<-1.6074944589803657,2.5630458855912037,-1.0938146616894917>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5 }
    cylinder {  m*<-2.268433341197093,-4.080269603430596,-1.4423664199807393>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5}

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
    sphere { m*<0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 1 }        
    sphere {  m*<0.25485600565421745,0.36328201935292714,2.867685874847031>, 1 }
    sphere {  m*<2.7488292949187874,0.3366059165589764,-1.3490784217247063>, 1 }
    sphere {  m*<-1.6074944589803657,2.5630458855912037,-1.0938146616894917>, 1}
    sphere { m*<-2.268433341197093,-4.080269603430596,-1.4423664199807393>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.25485600565421745,0.36328201935292714,2.867685874847031>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5 }
    cylinder { m*<2.7488292949187874,0.3366059165589764,-1.3490784217247063>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5}
    cylinder { m*<-1.6074944589803657,2.5630458855912037,-1.0938146616894917>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5 }
    cylinder {  m*<-2.268433341197093,-4.080269603430596,-1.4423664199807393>, <0.014120900912525713,0.23457194117260172,-0.11986889627352032>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    