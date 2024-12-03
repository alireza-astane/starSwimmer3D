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
    sphere { m*<0.4035786882077969,0.9707860514395887,0.10578048799278923>, 1 }        
    sphere {  m*<0.6443137929494887,1.0994961296199144,3.0933352591133407>, 1 }
    sphere {  m*<3.138287082214054,1.0728200268259633,-1.1234290374583948>, 1 }
    sphere {  m*<-1.218036671685093,3.29925999585819,-0.8681652774231811>, 1}
    sphere { m*<-3.654295519987101,-6.700043361188342,-2.2453262208200506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6443137929494887,1.0994961296199144,3.0933352591133407>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5 }
    cylinder { m*<3.138287082214054,1.0728200268259633,-1.1234290374583948>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5}
    cylinder { m*<-1.218036671685093,3.29925999585819,-0.8681652774231811>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5 }
    cylinder {  m*<-3.654295519987101,-6.700043361188342,-2.2453262208200506>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5}

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
    sphere { m*<0.4035786882077969,0.9707860514395887,0.10578048799278923>, 1 }        
    sphere {  m*<0.6443137929494887,1.0994961296199144,3.0933352591133407>, 1 }
    sphere {  m*<3.138287082214054,1.0728200268259633,-1.1234290374583948>, 1 }
    sphere {  m*<-1.218036671685093,3.29925999585819,-0.8681652774231811>, 1}
    sphere { m*<-3.654295519987101,-6.700043361188342,-2.2453262208200506>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6443137929494887,1.0994961296199144,3.0933352591133407>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5 }
    cylinder { m*<3.138287082214054,1.0728200268259633,-1.1234290374583948>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5}
    cylinder { m*<-1.218036671685093,3.29925999585819,-0.8681652774231811>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5 }
    cylinder {  m*<-3.654295519987101,-6.700043361188342,-2.2453262208200506>, <0.4035786882077969,0.9707860514395887,0.10578048799278923>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    