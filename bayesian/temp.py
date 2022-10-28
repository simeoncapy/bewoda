import random
 
# Utility function to find ceiling of r in arr[l..h]
def findCeil(arr, r, l, h) :
 
    while (l < h) :   
        mid = l + ((h - l) >> 1); # Same as mid = (l+h)/2
        if r > arr[mid] :
            l = mid + 1
        else :
            h = mid
     
    if arr[l] >= r :
        return l
    else :
        return -1
 
# The main function that returns a random number
# from arr[] according to distribution array
# defined by freq[]. n is size of arrays.
def myRand(arr, freq, n) :
 
    # Create and fill prefix array
    prefix = [0] * n
    prefix[0] = freq[0]
    for i in range(n) :
        prefix[i] = prefix[i - 1] + freq[i]

    print(prefix)
 
    # prefix[n-1] is sum of all frequencies.
    # Generate a random number with
    # value from 1 to this sum
    r = random.randint(0, prefix[n - 1]) + 1
 
    # Find index of ceiling of r in prefix array
    indexc = findCeil(prefix, r, 0, n - 1)
    return arr[indexc]
 
# Driver code
arr = [1, 2, 3, 4]
freq = [10, 5, 20, 100]
n = len(arr)
 
# Let us generate 10 random numbers according to
# given distribution
for i in range(5) :
    print(myRand(arr, freq, n))